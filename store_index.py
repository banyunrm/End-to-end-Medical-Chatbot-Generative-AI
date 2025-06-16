import os
from src.helper import (
    load_pdf_file,
    text_split,
    download_hugging_face_embeddings,
    build_faiss_index
)

# Set path directory data
project_dir = r"C:\Users\Lenovo ThinkPad X280\Downloads\End-to-end-Medical-Chatbot-Generative-AI"
data_dir = os.path.join(project_dir, 'Data')

# Check if the data directory is available
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} does not exist!")

# 1. Load PDF
extracted_data = load_pdf_file(data_path=data_dir)

# 2. Split documents
text_chunks = text_split(extracted_data)

# 3. Download embeddings
embeddings = download_hugging_face_embeddings()

# 4. Build dan simpan FAISS index
build_faiss_index(chunks=text_chunks, embeddings=embeddings, index_path="faiss_index")

print("✅ FAISS index berhasil disimpan di folder 'faiss_index'")
