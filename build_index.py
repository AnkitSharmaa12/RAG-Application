"""Chunk + embed the knowledge base and write the FAISS index to ./index."""
from rag import config
from rag.store import build_index

if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks from {config.KB_PATH.name} -> {config.INDEX_DIR}")
