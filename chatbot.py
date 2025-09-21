from flask import Blueprint, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import json
import os

chatbot_bp = Blueprint("chatbot_bp", __name__)

# Global placeholders
embedder = None
index = None
chunked_texts = None
gen_model = None

# Lazy loader function
def lazy_load():
    global embedder, index, chunked_texts, gen_model
    if embedder is None:
        # Load SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    if index is None or chunked_texts is None:
        # Load FAISS index and chunked texts
        index = faiss.read_index("faiss_index.index")
        with open("chunked_texts.json", "r", encoding="utf-8") as f:
            chunked_texts = json.load(f)
    if gen_model is None:
        # Configure Gemini
        GENIE_API_KEY = os.environ.get("GENIE_API_KEY")
        genai.configure(api_key=GENIE_API_KEY)
        gen_model = genai.GenerativeModel("gemini-1.5-flash")

# Function to get response from Gemini with context
def get_response(query: str) -> str:
    lazy_load()  # Ensure everything is loaded
    # Embed query
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    # FAISS search
    distances, indices = index.search(query_embedding, k=3)
    context = "\n\n".join([chunked_texts[idx] for idx in indices[0]])
    # Prepare prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in detail with two paragraphs,give therapy suggestions in second paragraph and continue chatting naturally."
    # Gemini API call
    response = gen_model.generate_content(prompt)
    return response.text

@chatbot_bp.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400
    query = data["query"]
    answer = get_response(query)
    return jsonify({"answer": answer})
