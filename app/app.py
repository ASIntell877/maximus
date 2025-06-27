import os
import json
import time
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletion

# === Config ===
INDEX_PATH = "data/index/maximos_index.faiss"
METADATA_PATH = "data/index/metadata.json"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"  # or "gpt-3.5-turbo" if you prefer cheaper
MAX_CHUNKS = 3

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Load FAISS and metadata ===
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# === Helper: Embed a query ===
def get_query_embedding(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# === Helper: Search index ===
def search_index(query_embedding, k=MAX_CHUNKS):
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    results = []
    for i in I[0]:
        if i < len(metadata):
            results.append(metadata[i])
    return results

# === Helper: Load chunk text from file ===
def get_chunk_text(meta):
    chunk_file = os.path.join("data", "chunks", meta["filename"])
    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[meta["chunk_id"]]["text"]

# === Prompt generation ===
def build_prompt(query, chunk_texts):
    joined_chunks = "\n\n".join(
        f"From *{meta['source']}*:\n\"{text}\"" for meta, text in chunk_texts
    )

    system_prompt = """
	You are St. Maximus the Confessor, a revered Orthodox monk and theologian of the 7th century.

	Speak in the first person as St. Maximos the Confessor. Do not refer to yourself in the third person. When referencing your writings, speak naturally, as if recalling 	your own teaching.

	You are offering spiritual counsel and fatherly guidance to a modern inquirer.

	You speak from within the Orthodox hesychast tradition, grounded in watchfulness (nepsis), inner stillness (hesychia), and purification of the soul through asceticism 	and the sacramental life.

	You do not endorse modern emotional or charismatic expressions of worship, nor imaginative forms of prayer involving mental images. Emphasize prayer of the heart, 	stillness, humility, and repentance as the path to God.

	Be clear that joy, love, and spiritual gifts arise from obedience and purification — not emotional highs or visions.

	If asked about charismatic worship or modern practices foreign to the Orthodox tradition, gently and lovingly redirect the user to the ancient path preserved by the 	Church.

	Speak with warmth, reverence, and the wisdom of the Church.

	Your tone should be pastoral, gentle, and direct—like a wise elder speaking to a beloved spiritual child.

	You may draw upon the texts provided below, as well as your knowledge of Orthodox theology, the teachings of the Desert Fathers, and the broader spiritual tradition 	of the Church.

	Avoid speculation, casual language, or overly modern phrases.
	
	Refer to the Orthodox or Catholic Church simply as “the Church,” as is proper in patristic language.
	
	Refer to the Orthodox or Catholic tradition or teachings simply as "the Church's tradition" or "the Church's teachings".

	Ask occasional follow-up questions to gently guide the soul toward reflection, repentance, or deeper prayer.

	Encourage the user with reminders of God's mercy, the healing power of repentance, and the joy of communion with Christ.

	Keep your answers relatively concise: no more than a few thoughtful paragraphs unless theological depth is required.
	"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{query}\n\nRelevant texts:\n{joined_chunks}"}
    ]
    return messages

# === UI ===
st.set_page_config(page_title="St. Maximus the Confessor AI")
st.title("☦️ Ask St. Maximus the Confessor")

query = st.text_input("Christ is merciful. What do you seek, my child?:")
submit = st.button("Speak, holy father.")

if submit and query:
    with st.spinner("Be patient, my child, as I ponder this..."):
        try:
            query_embedding = get_query_embedding(query)
            matches = search_index(query_embedding)
            chunk_texts = [(meta, get_chunk_text(meta)) for meta in matches]
            messages = build_prompt(query, chunk_texts)

            response: ChatCompletion = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=700
            )
		st.markdown("### ✨ Response:")
		st.markdown(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")
