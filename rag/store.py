"""Build / save / load the FAISS index and its chunk metadata."""
import json

import faiss
import numpy as np

from . import config, embedder
from .chunker import Chunk, chunk_markdown


def build_index(client=None) -> int:
    """Chunk the knowledge base, embed it, and write the index to disk. Returns chunk count."""
    md = config.KB_PATH.read_text(encoding="utf-8")
    chunks = chunk_markdown(md)
    vectors = embedder.embed_documents([c.text for c in chunks], client)

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors == cosine
    index.add(vectors)

    config.INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(config.INDEX_FILE))
    config.CHUNKS_FILE.write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(chunks)


class VectorStore:
    def __init__(self):
        if not config.INDEX_FILE.exists() or not config.CHUNKS_FILE.exists():
            raise FileNotFoundError("Index not found. Run `python build_index.py` first.")
        self.index = faiss.read_index(str(config.INDEX_FILE))
        raw = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in raw]

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        scores, ids = self.index.search(query_vec, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
