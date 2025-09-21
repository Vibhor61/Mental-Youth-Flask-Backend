from flask import Blueprint, request, jsonify
import numpy as np
import faiss
import google.generativeai as genai
import json
import os

chatbot_bp = Blueprint("chatbot_bp", __name__)

embeddings = np.load("ChunkedEmbedding.npy").astype(np.float32)
index = faiss.read_index("faiss_index.index")
with open("chunked_texts.json", "r", encoding="utf-8") as f:
    chunked_texts = json.load(f)

GENIE_API_KEY = os.environ.get("GENIE_API_KEY")
genai.configure(api_key=GENIE_API_KEY)
gen_model = genai.GenerativeModel("gemini-1.5-flash")

def get_response(query: str) -> str:
    query_embedding = embeddings[0:1]  # placeholder
    distances, indices = index.search(query_embedding, k=3)
    context = "\n\n".join([chunked_texts[idx] for idx in indices[0]])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in brief and continue chatting naturally."
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
