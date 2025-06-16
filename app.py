# Import Libraryfrom flask import Flask, render_template, request
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
from flask import Flask, render_template, request

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

import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()