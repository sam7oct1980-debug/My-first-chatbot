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
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pptx import Presentation
from docx import Document
import glob

# **1️⃣ Set Up API Keys & Environment**
os.environ["GOOGLE_API_KEY"] = "AIzaSyBMx_ZelxjCy6zNnaaArj78xd1rx8VWTdA"

# **2️⃣ Authenticate Google Drive API (For Cloud Use)**
def authenticate_google_drive():
    creds = service_account.Credentials.from_service_account_file(
        "service_account.json", scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

drive_service = authenticate_google_drive()

# **3️⃣ Define Folder Path (Google Drive or Local)**
FOLDER_ID = "your_google_drive_folder_id_here"  # Replace with actual folder ID

# **4️⃣ Function to List Google Drive Files**
def list_drive_files():
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents",
        fields="files(id, name)"
    ).execute()
    return results.get("files", [])

# **5️⃣ Download Files from Google Drive**
def download_drive_file(file_id, file_name):
    file_path = os.path.join("temp_files", file_name)
    os.makedirs("temp_files", exist_ok=True)
    
    request = drive_service.files().get_media(fileId=file_id)
    with open(file_path, "wb") as f:
        f.write(request.execute())
    return file_path

# **6️⃣ Extract Text from Different Document Types**
def extract_text_from_file(file_path):
    text = ""

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif file_path.endswith(".pptx"):
        presentation = Presentation(file_path)
        text_list = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_list.append(shape.text.strip())
        text = "\n".join(text_list)

    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        text = df.to_string()


    return text.strip()

# **7️⃣ Load and Process Documents**
def load_and_process_documents():
    files = list_drive_files()
    all_texts = []
    
    if not files:
        raise ValueError("⚠️ No supported files found in the specified Google Drive folder!")

    for file in files:
        file_path = download_drive_file(file["id"], file["name"])
        extracted_text = extract_text_from_file(file_path)
        if extracted_text:
            all_texts.append(extracted_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(all_texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# **8️⃣ Load Documents at Startup**
vectorstore = load_and_process_documents()

# **9️⃣ AI-Powered Question Answering**
def ask_question(query):
    if not vectorstore:
        return "⚠️ No documents available. Please check the Drive folder and restart the application."

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    return qa.run(query)

# **🔹 10️⃣ Streamlit UI**
st.markdown("<h2 style='text-align: center; color: blue;'>✦ ⚙️ AI-Powered Course Chatbot ✦</h2>", unsafe_allow_html=True)
st.write("This chatbot retrieves answers from course documents stored in Google Drive.")

query = st.text_input("Ask a question:")
if st.button("💬 Get Answer"):
    if query:
        response = ask_question(query)
        st.write(f"**AI:** {response}")
    else:
        st.warning("Please enter a question.")
