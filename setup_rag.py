import os
import pickle
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_DIR = "rag/pdfs"
STORE_DIR = "rag/vectorstore"


def read_all_pdfs():
    """Read all PDFs in the folder and return combined text."""
    all_text = ""
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            print(f"Reading {filename}...")
            reader = PdfReader(os.path.join(PDF_DIR, filename))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n"
    return all_text


def chunk_text(text, chunk_size=500, overlap=50):
    """Break text into small overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_index():
    print("Reading PDFs...")
    text = read_all_pdfs()

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")
    vectors = embedder.encode(chunks, show_progress_bar=True)

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    os.makedirs(STORE_DIR, exist_ok=True)
    faiss.write_index(index, f"{STORE_DIR}/index.faiss")
    with open(f"{STORE_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Done! Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    build_index()