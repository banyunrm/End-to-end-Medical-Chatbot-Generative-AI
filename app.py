# Import Library
from flask import Flask, render_template, request
from dotenv import load_dotenv
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from src.helper import (
    download_hugging_face_embeddings,
    clean_text,
    load_faiss_index,
    generate_prompt,
    save_chat_history
)
import os
import gradio as gr

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Path FAISS index
INDEX_PATH = "faiss_index"

# 1. Load Embeddings & FAISS
embeddings = download_hugging_face_embeddings()
retriever = load_faiss_index(INDEX_PATH, embeddings, top_k=5)

# 2. Load LLM pipeline
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", device=0)

# 3. System Prompt
SYSTEM_PROMPT = (
    "You are a medical assistant specialized in dermatology. "
    "Using only the context provided below, answer the user question precisely. "
    "Do not use prior knowledge or guess. If the answer is not in the context, reply: "
    "'Sorry, I couldn't find the answer in the documents.' Answer in maximum 3 short sentences."
    "\n\nContext:\n{context}\n\nQuestion: {input}"
)

# 4. Chat History
chat_history = []

# 5. RAG-style QA Chain
def rag_chain(user_input):
    docs = retriever.invoke(user_input)
    context = "\n\n".join([clean_text(doc.page_content) for doc in docs])
    prompt = generate_prompt(SYSTEM_PROMPT, context, user_input)
    result = qa_pipeline(prompt, max_new_tokens=256, do_sample=False)
    final_answer = result[0]['generated_text']
    save_chat_history(chat_history, user_input, final_answer)
    return final_answer

# Gradio Interface
def chat_interface(message, history):
    response = rag_chain(message)
    return response

# Create Gradio Interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="Medical Chatbot",
    description="Ask questions about dermatology and get accurate answers based on medical documents.",
    theme="soft",
    examples=[
        "What is acne?",
        "How to treat eczema?",
        "What are the symptoms of psoriasis?"
    ]
)

# Run both Flask and Gradio
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)

