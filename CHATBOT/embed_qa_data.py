import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
with open("inara_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

questions = [item["q"] for item in qa_data]
answers = [item["a"] for item in qa_data]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(questions, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "inara_index.faiss")
with open("inara_metadata.json", "w", encoding="utf-8") as f:
    json.dump(qa_data, f, indent=2)
