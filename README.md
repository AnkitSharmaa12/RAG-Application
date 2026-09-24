# MecroTech RAG Assistant

Streamlit chatbot that answers questions about MecroTech using only `mecrotech-rag-knowledge-base.md`.

- **LLM:** `gemini-flash-lite-latest` · **Embeddings:** `gemini-embedding-001` (768-dim) · **Vector DB:** FAISS (`IndexFlatIP`, cosine)

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env: GEMINI_API_KEY=...
python build_index.py       # chunk + embed the knowledge base -> ./index
streamlit run app.py
```

## How it works
1. `rag/chunker.py` splits the markdown by `##`/`###` heading; each chunk is prefixed with its heading path.
2. `rag/embedder.py` embeds chunks (`RETRIEVAL_DOCUMENT`) and questions (`RETRIEVAL_QUERY`), L2-normalised.
3. `rag/store.py` stores vectors in FAISS and chunk text in `index/chunks.json`.
4. `rag/pipeline.py`: rewrites follow-ups into standalone questions, retrieves top-k, drops chunks below
   `MIN_SCORE`, and asks Gemini to answer from that context only. If nothing relevant is found the LLM is skipped
   and a "not found" message with the company contact is returned.
5. `app.py` is the chat UI (streaming answers, sources expander, suggestions, sidebar controls).

## Updating the knowledge base
Edit `mecrotech-rag-knowledge-base.md`, then click **Rebuild index** in the sidebar (or run `python build_index.py`).

## Testing
```bash
python evaluate.py   # golden questions, refusal checks, follow-up check; prints top similarity scores
```
Tune `MIN_SCORE`, `TOP_K` and chunk sizes in `rag/config.py`.
