import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from docx import Document
from PyPDF2 import PdfReader

# **1️⃣ Set Up API Key (Ensure secrets.toml is configured)**
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# **2️⃣ File Upload Handling**
st.title("📚 RAG Chatbot for Course Documents")
st.write("Upload PDF, DOCX, TXT, or CSV files to retrieve information.")

uploaded_files = st.file_uploader(
    "Upload your documents", type=["pdf", "docx", "txt", "csv", "xlsx"], accept_multiple_files=True
)

# **3️⃣ Function to Extract Text from Different File Types**
def extract_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    elif file.name.endswith(".docx"):
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")

    elif file.name.endswith(".csv") or file.name.endswith(".xlsx"):
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        text = df.to_string()

    return text.strip()

# **4️⃣ Process Uploaded Files**
all_texts = []
if uploaded_files:
    for file in uploaded_files:
        extracted_text = extract_text_from_file(file)
        if extracted_text:
            all_texts.append(extracted_text)

# **5️⃣ Chunking & Vectorization**
if all_texts:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(all_texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # **6️⃣ AI-Powered Retrieval**
    def ask_question(query):
        if not vectorstore:
            return "⚠️ No documents available. Please upload and process files."

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever())
        return qa.run(query)

    # **7️⃣ Streamlit Chat Interface**
    query = st.text_input("Ask a question:")
    if st.button("💬 Get Answer"):
        if query:
            response = ask_question(query)
            st.write(f"**AI:** {response}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("📂 Please upload some documents to get started.")
