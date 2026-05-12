import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from modules.llm import chat

STORE_DIR = "rag/vectorstore"
# L2 distance threshold — below this means the chunk is relevant to the query.
# With normalized sentence-transformer embeddings, distances range 0-2.
# 1.0 ≈ cosine similarity of 0.5, a reasonable relevance cutoff.
RELEVANCE_THRESHOLD = 1.0

print("Loading embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedder loaded.")

_index = None
_chunks = None


def load_vectorstore():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index(f"{STORE_DIR}/index.faiss")
        with open(f"{STORE_DIR}/chunks.pkl", "rb") as f:
            _chunks = pickle.load(f)
    return _index, _chunks


def retrieve(query, k=3):
    index, chunks = load_vectorstore()
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [chunks[i] for i in indices[0]], distances[0]


def answer_query(query):
    if not os.path.exists(f"{STORE_DIR}/index.faiss"):
        return _general_answer(query)

    relevant_chunks, distances = retrieve(query, k=3)

    if distances[0] <= RELEVANCE_THRESHOLD:
        context = "\n\n".join(relevant_chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful medical assistant. Answer the user's question "
                    "using the context below. Be concise and clear.\n\n"
                    f"CONTEXT:\n{context}"
                )
            },
            {"role": "user", "content": query}
        ]
        return chat(messages, max_tokens=200)

    return _general_answer(query)


def _general_answer(query):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical assistant. Answer the user's health-related "
                "question using your general medical knowledge. Be concise and practical. "
                "If symptoms sound serious, advise them to see a doctor."
            )
        },
        {"role": "user", "content": query}
    ]
    return chat(messages, max_tokens=200)
