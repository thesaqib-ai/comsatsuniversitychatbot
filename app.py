import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def main():
    load_dotenv()  # Load environment variables from .env file
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")
    
    # Streamlit Page Configuration
    page_icon = "images/comsats icon.jpg"
    image = "images/comsats.png"
    st.set_page_config(page_title="COMSATS", page_icon=page_icon, layout="wide")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(image, width=150, use_column_width=100)
    with col2:
        st.title("COMSATS University Islamabad\nAsk any query related to academic programs, admissions, fee structure, campus life etc.")
    
    # Initialize chat session in Streamlit if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize Qdrant in session state if not already present
    if "qdrant" not in st.session_state:
        st.session_state.qdrant = get_qdrant()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input field for user's message
    query_text = st.chat_input()

    # Check if there's a query and if Qdrant is available
    if query_text:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(query_text)
        st.session_state.chat_history.append({"role": "user", "content": query_text})
        
        # Generate response from the model
        with st.spinner('Thinking...'):
            try:
                retriever = st.session_state.qdrant.as_retriever()
                response = generate_response(retriever, query_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')

def get_qdrant():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name="COMSATS Embeddings - VectorStore",
        url="https://f6c816ad-c10a-4487-9692-88d5ee23882a.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key=os.getenv("QDRANT_API_KEY")
    )
    return qdrant

def generate_response(retriever, query_text):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=1024,
        max_retries=2
    )

    template = """Use the following content of Comsats University Islamabad to answer the question at the end. Go through the content and look for the answers.
    If you don't find relevant information in the content, just ask the user to ask relevant questions.!, Don't try to make up an answer.
    Give the answer in detail. Note that you can reply to greetings!

    {context}

    Question: {question}

    Helpful Answer:"""

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query_text)

if __name__ == "__main__":
    main()
