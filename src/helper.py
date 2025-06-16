import os
import re
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#  1. Load PDF Files 
def load_pdf_file(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

#  2. Split Documents 
def text_split(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

#  3. Download Embedding Model 
def download_hugging_face_embeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    return HuggingFaceEmbeddings(model_name=model_name)

#  4. Build FAISS Index 
def build_faiss_index(chunks, embeddings, index_path="faiss_index"):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

#  5. Load FAISS Index 
def load_faiss_index(index_path, embeddings, top_k=5):
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

#  6. Text Cleaning 
def clean_text(text):
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text

#  7. Generate Prompt 
def generate_prompt(template, context, user_input):
    return template.replace("{context}", context).replace("{input}", user_input)

#  8. Save Chat History 
def save_chat_history(chat_history, user_input, response):
    chat_history.append({
        "question": user_input,
        "answer": response
    })
