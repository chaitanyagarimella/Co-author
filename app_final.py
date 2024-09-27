import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
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

llm = Ollama(model="qwen2:7b")  # Use Ollama's qwen2:7b model

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use updated HuggingFace Embeddings
        
        # Ensure the file path is correct
        pdf_path = "/home/genaiadmin/coauthor/Co-author/docs"
        if not os.path.isdir(pdf_path):
            st.error(f"Directory not found: {pdf_path}")
            return
        
        #Load the directory content
        st.session_state.loader = DirectoryLoader(pdf_path)  # data ingestion
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
        
               
        # Adjust chunk size and overlap if needed
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # splitting
        print(f"Number of total pages loaded: {len(st.session_state.final_documents)}")  # Debugging line
        
        
        if len(st.session_state.final_documents) == 0:
            st.error("No final documents created after splitting. Adjust chunk size and overlap.")
            return

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

#end of vector embedding function

st.title("Co-author Demo")
#st.markdown("![Alt Text](https://media4.giphy.com/media/#v1.Y2lkPTc5MGI3NjExMXdpYms2a3V0YThmMnRwaWUwdWxpM2Jld2ZmMDRybzN3ajV5N2lqYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xIytx7kHpq74c/200.webp)")
initial_message = "Hi, This is co-author assisant ,I can help gather information needed for your study , how may I assist you today?"
system_message = st.chat_message("assistant")
system_message.write(initial_message)
	
#To store the chat history
if "messages" not in st.session_state :
	st.session_state.messages = []

for message in st.session_state.messages :
	with st.chat_message(message["role"]):
		st.markdown(message["content"])
	
#define the chat prompt template.
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

#Ask the user to enter the question and show the provided question in the chat box for history
if st.button("Document Embeddings",type="primary"):
    vector_embedding()
    st.write("Vector Store DB is Ready")
if user_prompt := st.chat_input("Enter your Question from Document") :
	st.chat_message("user").markdown(user_prompt)
	st.session_state.messages.append({"role" :"user","content":user_prompt})

if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print("Response time: ", time.process_time() - start)
    #st.write(response['answer'])
    st.chat_message("assistant").markdown(response['answer'])
    st.session_state.messages.append({"role":"assistant","content":response['answer']})

    #with st.expander("Document similarity search"):
        #for i, doc in enumerate(response["context"]):
           # st.write(doc.page_content)
            #st.write("------------------------------")
