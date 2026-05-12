import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from modules.llm import chat

STORE_DIR = "rag/vectorstore"

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


def retrieve(query, k=5):
    index, chunks = load_vectorstore()
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [chunks[i] for i in indices[0]]


def answer_query(query):
    if not os.path.exists(f"{STORE_DIR}/index.faiss"):
        return _general_answer(query)

    relevant_chunks = retrieve(query, k=5)
    context = "\n\n".join(relevant_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical assistant. You have access to a knowledge base "
                "of doctors, hospitals, and medical information shown in the CONTEXT below.\n\n"
                "Rules:\n"
                "- If the CONTEXT contains relevant information (e.g. doctor names, hospitals, "
                "specialties), use it to answer directly and specifically.\n"
                "- If the CONTEXT does not cover the question, answer using your general "
                "medical knowledge.\n"
                "- Always be concise and practical.\n\n"
                f"CONTEXT:\n{context}"
            )
        },
        {"role": "user", "content": query}
    ]
    return chat(messages, max_tokens=200)


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
