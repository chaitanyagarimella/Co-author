import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from dotenv import load_dotenv
import time

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = Ollama(model="qwen:0.5b")  # Use Ollama's qwen:.5b model

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use updated HuggingFace Embeddings
        
        # Ensure the file path is correct
        pdf_path = "./The_Archer.pdf"
        if not os.path.isfile(pdf_path):
            st.error(f"File not found: {pdf_path}")
            return
        
        st.session_state.loader = PyPDFLoader(pdf_path)  # data ingestion
        try:
            st.session_state.docs = st.session_state.loader.load()  # document loading
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return
        
        # Debugging lines to check the loaded documents
        print(f"Number of documents loaded: {len(st.session_state.docs)}")
        if len(st.session_state.docs) == 0:
            st.error("No documents were loaded. Check the file path and format.")
            return
        
        # Print the content of the loaded documents
        for doc in st.session_state.docs:
            print(doc.page_content[:500])  # Print the first 500 characters of each document for inspection
        
        # Adjust chunk size and overlap if needed
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # splitting
        print(f"Number of final documents: {len(st.session_state.final_documents)}")  # Debugging line
        
        # Print the content of the final split documents
        for final_doc in st.session_state.final_documents:
            print(final_doc.page_content[:500])  # Print the first 500 characters of each split document for inspection
        
        if len(st.session_state.final_documents) == 0:
            st.error("No final documents created after splitting. Adjust chunk size and overlap.")
            return

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Ollama qwen:0.5b Demo")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Question:{input}
"""
)

prompt1 = st.text_input("Enter your Question from Document")

if st.button("Document Embeddings"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time: ", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------------")
