"""Streamlit chat UI for the MecroTech RAG assistant."""
import streamlit as st

from rag import config
from rag.pipeline import RagPipeline
from rag.store import build_index

st.set_page_config(page_title="MecroTech Assistant", page_icon="💬", layout="centered")

SUGGESTIONS = [
    "What does MecroTech do?",
    "How much does MVP development cost?",
    "How long does it take to launch an MVP?",
    "What is the refund policy?",
    "How does the referral program work?",
    "How can I contact MecroTech?",
]


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_pipeline() -> RagPipeline:
    return RagPipeline()


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s.heading.replace('MecroTech > ', '')}**  · similarity {s.score:.2f}")
            st.caption(s.text.split("\n", 1)[-1][:400] + ("..." if len(s.text) > 400 else ""))


# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("MecroTech Assistant")
    st.write(
        "Ask anything about MecroTech: services, pricing, process, team, policies and the referral program. "
        "Answers come only from the company's website information."
    )
    top_k = st.slider("Passages to retrieve", 1, 10, config.TOP_K)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("Rebuild index", use_container_width=True, help="Re-embed the knowledge base file"):
        try:
            with st.spinner("Re-embedding knowledge base..."):
                n = build_index()
            load_pipeline.clear()
            st.success(f"Indexed {n} chunks.")
        except Exception as e:  # noqa: BLE001 - show any API/config error to the user
            st.error(str(e))
    st.caption(f"LLM: `{config.LLM_MODEL}` · Embeddings: `{config.EMBED_MODEL}` · FAISS")

# ---- main ------------------------------------------------------------------
st.title("💬 MecroTech Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    pipeline = load_pipeline()
except (RuntimeError, FileNotFoundError) as e:
    st.error(str(e))
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 2].button(q, key=f"sugg{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

question = st.chat_input("Ask about MecroTech...") or st.session_state.pop("pending", None)

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching..."):
                stream, sources = pipeline.answer_stream(question, history, top_k)
            answer = st.write_stream(stream)
            render_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:  # noqa: BLE001 - surface API errors (quota, bad key) instead of crashing
            st.error(f"Something went wrong: {e}")
            st.session_state.messages.pop()  # drop the unanswered question so history stays consistent
