import os
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from openai.types import CreateEmbeddingResponse

# === Config ===
CHUNKS_PATH = r"C:\Maximos\data\chunks"
INDEX_OUTPUT_PATH = r"C:\Maximos\data\index"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Optional: Hardcode here for testing if needed
# client = OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Make sure output path exists
os.makedirs(INDEX_OUTPUT_PATH, exist_ok=True)

texts = []
metadatas = []

# === Load all chunked JSON files ===
for filename in os.listdir(CHUNKS_PATH):
    if filename.endswith(".json"):
        full_path = os.path.join(CHUNKS_PATH, filename)
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                texts.append(item['text'])
                metadatas.append({
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "filename": filename
                })

print(f"Loaded {len(texts)} chunks from: {CHUNKS_PATH}")

# === Function to get embedding from OpenAI ===
def get_embedding(text, model=EMBEDDING_MODEL, retries=3):
    for attempt in range(retries):
        try:
            response: CreateEmbeddingResponse = client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error (attempt {attempt + 1}): {e}")
            time.sleep(5)

# === Generate embeddings for all texts ===
print("Generating embeddings via OpenAI...")

embeddings = []
for text in tqdm(texts):
    embedding = get_embedding(text)
    if embedding:
        embeddings.append(embedding)
    else:
        print("❌ Failed to embed a chunk. Skipping.")

# === Convert to FAISS index ===
print("Building FAISS index...")
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# === Save index and metadata ===
faiss.write_index(index, os.path.join(INDEX_OUTPUT_PATH, "maximos_index.faiss"))

with open(os.path.join(INDEX_OUTPUT_PATH, "metadata.json"), 'w', encoding='utf-8') as f:
    json.dump(metadatas, f, indent=2, ensure_ascii=False)

print("✅ Embeddings + FAISS index saved to:", INDEX_OUTPUT_PATH)
