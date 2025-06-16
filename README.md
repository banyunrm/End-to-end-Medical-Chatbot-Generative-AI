---
title: Medical Chatbot
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Medical Chatbot with RAG

A medical chatbot specialized in dermatology that uses Retrieval Augmented Generation (RAG) to provide accurate answers based on medical documents.

## Features

- Question answering based on medical documents
- Chat interface with example questions
- Accurate medical information from verified sources
- Powered by LangChain and Hugging Face Transformers

## How to Use

1. Type your medical question in the chat input
2. Click on example questions to try them out
3. Get instant, accurate answers based on medical documents

## Tech Stack

- Gradio (UI Framework)
- LangChain (RAG Implementation)
- FAISS (Vector Database)
- Hugging Face Transformers (LLM)

## Model Information

- Base Model: google/flan-t5-large
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS

## License

MIT License

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `Data` directory and add your PDF documents
4. Run the index creation script:
   ```bash
   python store_index.py
   ```
5. Start the application:
   ```bash
   python app.py
   ```

## Usage

1. Open your browser and navigate to `http://localhost:8080`
2. Type your medical question in the input field
3. Use voice input by clicking the microphone button
4. Toggle dark/light mode using the mode button
5. Save conversations using the save button

## Deployment

This application is configured for deployment on Hugging Face Spaces using Docker.

