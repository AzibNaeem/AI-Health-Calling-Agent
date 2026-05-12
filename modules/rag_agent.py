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


def load_vectorstore():
    """Load FAISS index and text chunks from disk."""
    index = faiss.read_index(f"{STORE_DIR}/index.faiss")
    with open(f"{STORE_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(query, k=3):
    """Find top-k most relevant text chunks for the query."""
    index, chunks = load_vectorstore()
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [chunks[i] for i in indices[0]]


def answer_query(query):
    """Retrieve context from PDFs and ask LLM to answer."""
    if not os.path.exists(f"{STORE_DIR}/index.faiss"):
        return "No PDF knowledge base found. Please run setup_rag.py first."

    relevant_chunks = retrieve(query, k=3)
    context = "\n\n".join(relevant_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful medical assistant. Answer the user's question "
                f"using ONLY the context below. Keep answer short and clear. "
                f"If the answer is not in the context, say you don't know.\n\n"
                f"CONTEXT:\n{context}"
            )
        },
        {"role": "user", "content": query}
    ]
    return chat(messages, max_tokens=200)