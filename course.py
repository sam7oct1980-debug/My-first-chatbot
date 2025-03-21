# -*- coding: utf-8 -*-
"""Course.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hzxzBWvd2Jg5eoNTvuDHYlIJ-O7qFpDK
"""

import os
import google.generativeai as genai
import gradio as gr
import glob
import pandas as pd
from docx import Document
from pptx import Presentation
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Set API Key
os.environ['GOOGLE_API_KEY'] = "AIzaSyBMx_ZelxjCy6zNnaaArj78xd1rx8VWTdA"

# Define the folder path containing documents (Modify as per your Drive structure)
DOCUMENTS_FOLDER_PATH = "D:/Agentic AI Assignment/"

vectorstore_path = "D:/Agentic AI Assignment/vectorstore"  # Modify this path as needed

# Function to extract text from different document types
def extract_text_from_file(file_path):
    """Extracts text from PDF, DOCX, PPTX, and XLSX files."""
    text = ""

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif file_path.endswith(".ppt") or file_path.endswith(".pptx"):
        presentation = Presentation(file_path)
        text_list = []
        for slide in presentation.slides:
          for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():  # Ensure shape has text
                text_list.append(shape.text.strip())  # Append clean text
        text = "\n".join(text_list)

    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        text = df.to_string()  # Convert entire spreadsheet to string

    return text.strip()

# Function to load and process all supported files from the given folder
def load_and_process_documents(folder_path):
    """Loads and processes all PDFs, DOCX, PPTX, and XLSX files from the folder."""
    all_texts = []
    supported_formats = ["pdf", "docx", "ppt", "pptx", "xls", "xlsx"]
    files = sum([glob.glob(f"{folder_path}/**/*.{ext}", recursive=True) for ext in supported_formats], [])

    if not files:
        raise ValueError("⚠️ No supported files found in the specified folder!")

    for file in files:
        extracted_text = extract_text_from_file(file)
        if extracted_text:
            all_texts.append(extracted_text)

    # Split documents into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(all_texts)

    # Create vector embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# Load and process documents at startup
vectorstore = load_and_process_documents(DOCUMENTS_FOLDER_PATH)

# Function to handle user queries
def ask_question(query):
    if not vectorstore:
        return "⚠️ No documents available. Please check the Drive folder and restart the application."

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    answer = qa.run(query)
    return f"「 ✦ ⚙️AI ✦ 」: {answer}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("<h2 style='text-align: center;  color: blue;'># 「 ✦ ⚙️AI-Powered Course Management ✦ 」</h2>")
    gr.Markdown("This system is preloaded with PDF, Word, Powerpoint and Excel files regarding Human Computer Interaction course from Google Drive. Ask any question based on them.")

    query_input = gr.Textbox(label="Ask a question")
    submit_button = gr.Button("💬 Get Answer")
    output_text = gr.Textbox(label="AI Response", interactive=False)

    submit_button.click(ask_question, inputs=[query_input], outputs=[output_text])

# Launch the interface
demo.launch()