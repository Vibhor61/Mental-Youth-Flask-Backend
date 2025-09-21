from flask import Blueprint, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import json
from dotenv import load_dotenv
import os

load_dotenv()
GENIE_API_KEY = os.getenv("GENIE_API_KEY")
genai.configure(api_key=GENIE_API_KEY)
gen_model = genai.GenerativeModel("gemini-1.5-flash")

# Load SentenceTransformer model (define globally)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and chunked texts from local files
def load_index_and_texts():
    faiss_index = faiss.read_index("faiss_index.index")
    with open("chunked_texts.json", "r", encoding="utf-8") as f:
        chunked_texts = json.load(f)
    return faiss_index, chunked_texts

# Initialize index and texts globally
index, chunked_texts = load_index_and_texts()

# Function to get response from Gemini with context
def get_response(query: str) -> str:
    # Embed the query
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Search FAISS index
    distances, indices = index.search(query_embedding, k=3)
    context = "\n\n".join([chunked_texts[idx] for idx in indices[0]])

    # Prepare prompt for Gemini
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in detail with two paragraphs,give thearapy suggestions in second paragraph and continue chatting naturally."

    # Call Gemini API
    response = gen_model.generate_content(prompt)
    return response.text

# Optional console chatbot
def chatbot():
    print("Hey there! Hope you're doing well today. What's on your mind? Say 'bye' to exit.\n")
    while True:
        query = input("You: ").strip()
        if "bye" in query.lower():
            print("Chatbot: See you later!")
            break
        answer = get_response(query)
        print(f"Chatbot: {answer}\n")

# Flask Blueprint for API
chatbot_bp = Blueprint("chatbot_bp", __name__)

@chatbot_bp.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    query = data["query"]
    answer = get_response(query)
    return jsonify({"answer": answer})

