import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

with open("inara_metadata.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

questions = [item["q"] for item in qa_data]
answers = [item["a"] for item in qa_data]

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("inara_index.faiss")

# Embed all questions (optional if already stored)
question_embeddings = model.encode(questions, convert_to_numpy=True)

print("🔍 Inara Technologies Chatbot (type 'exit' to quit)")
while True:
    query = input("\n🧠 You: ")
    if query.lower() in ["exit", "quit"]:
        break

    query_vec = model.encode([query], convert_to_numpy=True)

    # Search for top 1 most similar question
    D, I = index.search(query_vec, k=1)
    top_idx = I[0][0]
    confidence = 1 - D[0][0]  # cosine distance → similarity

    print(f"\n🤖 InaraBot (score: {confidence:.2f}): {answers[top_idx]}")
