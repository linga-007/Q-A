import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time
import tempfile

load_dotenv()

groq_api_key = "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"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model PDF Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

def vector_embedding(pdf_file):
    if "vectors" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_pdf_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:20])  
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectors = FAISS.from_documents(final_documents, embeddings)
        
        st.session_state.vectors = vectors
        st.session_state.final_documents = final_documents
        st.write("Vector Store DB Is Ready")

pdf_file = st.file_uploader("Upload a PDF document", type="pdf")

if pdf_file:
    if st.button("Generate Document Embedding"):
        vector_embedding(pdf_file)

if "vectors" in st.session_state:
    prompt1 = st.text_input("Enter Your Question From Documents")

    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)

        if 'answer' in response:
            st.write(response['answer'])
        else:
            st.error("No answer found.")
