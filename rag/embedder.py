"""Gemini embedding calls (batched, L2-normalised so inner product == cosine)."""
import time

import numpy as np
from google import genai
from google.genai import types

from . import config

_BATCH = 50


def get_client() -> genai.Client:
    key = config.get_api_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one at https://aistudio.google.com/apikey)."
        )
    return genai.Client(api_key=key)


def _normalise(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def _embed(texts: list[str], task_type: str, client: genai.Client | None = None) -> np.ndarray:
    client = client or get_client()
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=config.EMBED_DIM)
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        for attempt in range(4):
            try:
                resp = client.models.embed_content(model=config.EMBED_MODEL, contents=batch, config=cfg)
                out.extend(e.values for e in resp.embeddings)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    return _normalise(np.asarray(out, dtype="float32"))


def embed_documents(texts: list[str], client: genai.Client | None = None) -> np.ndarray:
    return _embed(texts, "RETRIEVAL_DOCUMENT", client)


def embed_query(text: str, client: genai.Client | None = None) -> np.ndarray:
    return _embed([text], "RETRIEVAL_QUERY", client)
