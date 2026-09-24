"""Central settings for the MecroTech RAG app."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KB_PATH = ROOT / "mecrotech-rag-knowledge-base.md"
INDEX_DIR = ROOT / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

# Models
LLM_MODEL = "gemini-flash-lite-latest"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # gemini-embedding-001 supports reduced dims; 768 is plenty here

# Chunking (characters; ~4 chars per token)
MAX_CHUNK_CHARS = 3200
CHUNK_OVERLAP_CHARS = 300

# Retrieval
TOP_K = 5
MIN_SCORE = 0.45  # cosine similarity below this is treated as "not in the docs"

# Generation
TEMPERATURE = 0.2
HISTORY_TURNS = 4  # previous user/assistant pairs sent for follow-up questions

COMPANY_CONTACT = "hello@mecro.tech or +91 70118 48978"


def get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
