import os
import re
import json
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Qiskit API reference  (offline: no LLM, no embeddings, no network)
# Reads the JSON produced by qiskit_reference.py -- {signature, docstring, ...}
# --------------------------------------------------------------------------- #
QISKIT_REFERENCE_PATH = os.path.join(HERE, "qiskit_reference.json")


@lru_cache(maxsize=1)
def load_qiskit_reference() -> dict:
    """Load the Qiskit reference JSON once (cached). Run qiskit_reference.py first."""
    if not os.path.exists(QISKIT_REFERENCE_PATH):
        raise FileNotFoundError(
            f"{QISKIT_REFERENCE_PATH} not found. Run `python qiskit_reference.py` first."
        )
    with open(QISKIT_REFERENCE_PATH, encoding="utf-8") as f:
        return json.load(f)


def search_qiskit_reference(query: str, top_k: int = 5, doc_chars: int = 600) -> str:
    """Keyword lookup over the Qiskit reference. No LLM -- just scores name/docstring
    matches and returns the top-k formatted `name(signature)\\ndocstring` blocks."""
    ref = load_qiskit_reference()
    terms = [t for t in re.split(r"\W+", query.lower()) if t]

    scored = []
    for name, entry in ref.items():
        short = name.rsplit(".", 1)[-1].lower()  # e.g. "zzfeaturemap"
        doc = entry.get("docstring", "").lower()
        score = 0
        for t in terms:
            if t == short:
                score += 10          # exact method-name hit
            elif t in short:
                score += 5           # partial name hit
            elif t in doc:
                score += 1           # docstring mention
        if score:
            scored.append((score, name, entry))

    if not scored:
        return f"No Qiskit reference entries matched: {query!r}"

    scored.sort(key=lambda x: x[0], reverse=True)
    blocks = []
    for _, name, entry in scored[:top_k]:
        sig = entry.get("signature") or ""
        doc = entry.get("docstring", "")
        if len(doc) > doc_chars:
            doc = doc[:doc_chars].rstrip() + " ..."
        blocks.append(f"{name}{sig}\n{doc}")
    return "\n\n---\n\n".join(blocks)


@tool
def qiskit_docs(query: str) -> str:
    """Look up Qiskit API documentation (signatures + docstrings) by method name or keyword,
    e.g. 'ZZFeatureMap', 'PauliFeatureMap', 'DensityMatrix'."""
    return search_qiskit_reference(query)


# --------------------------------------------------------------------------- #
# RAG over the local PDF knowledge base (Chroma)  -- lazy so import stays cheap
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="pdf_docs",
        embedding_function=embeddings,
        persist_directory=os.path.join(HERE, "chroma_db"),
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@tool
def rag_search(query: str) -> str:
    """Search the internal PDF knowledge base."""
    docs = get_retriever().invoke(query)
    return "\n\n".join(
        f"[{d.metadata.get('source')} p.{d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


# --------------------------------------------------------------------------- #
# Document-level RAG with a summarizer (the astronaut pattern): the chunk hit
# only says WHICH paper is relevant; the FULL paper is then loaded from disk
# and distilled by a cheap LLM into a query-focused summary. The agent never
# sees raw 1000-char fragments -- it sees a digest of the whole document.
# --------------------------------------------------------------------------- #
RAG_SUMMARY_MODEL = os.getenv("RAG_SUMMARY_MODEL", "openai:gpt-4o-mini")
MAX_DOC_CHARS = 60_000        # cap on full-paper text fed to the summarizer
MAX_SUMMARY_WORDS = 300


@lru_cache(maxsize=1)
def get_summary_llm():
    return init_chat_model(RAG_SUMMARY_MODEL, temperature=0)


def _resolve_pdf_path(source: str) -> str | None:
    """Chunk metadata stores the path ingest.py was given (e.g. 'pdfs/x.pdf',
    relative to this folder). Return an existing absolute path or None."""
    if not source:
        return None
    if os.path.isabs(source) and os.path.exists(source):
        return source
    candidate = os.path.join(HERE, source)
    return candidate if os.path.exists(candidate) else None


def _load_pdf_text(path: str) -> str:
    import fitz  # PyMuPDF (already required by the ingest loader)
    with fitz.open(path) as pdf:
        return "\n".join(page.get_text() for page in pdf)


def search_and_summarize_papers(query: str, max_docs: int = 3) -> str:
    """Retrieve chunks for `query`, map them back to their source PDFs
    (deduped, at most `max_docs` papers), load each FULL paper, and have a
    cheap LLM summarize it with respect to the query. Plain function so other
    code (e.g. a grounded expert) can reuse it without the tool wrapper."""
    docs = get_retriever().invoke(query)
    summarizer = get_summary_llm()
    seen: set = set()
    blocks = []
    for d in docs:
        src = d.metadata.get("source")
        if src in seen:
            continue
        seen.add(src)
        if len(blocks) >= max_docs:
            break
        path = _resolve_pdf_path(src)
        try:
            text = _load_pdf_text(path)[:MAX_DOC_CHARS] if path else d.page_content
        except Exception:
            text = d.page_content  # PDF unreadable -> at least the chunk
        try:
            msg = summarizer.invoke(
                f"Summarize the following paper in at most {MAX_SUMMARY_WORDS} "
                f"words, focusing ONLY on what is relevant to this query:\n"
                f"{query}\n\n"
                "Include concrete techniques, circuit structures, parameter "
                "choices, and reported results. If the paper is irrelevant to "
                "the query, say so in one line instead.\n\n"
                f"## Paper ({src})\n{text}")
            summary = msg.content if isinstance(msg.content, str) else str(msg.content)
        except Exception as exc:
            summary = f"(summarizer failed: {exc})\n{d.page_content}"
        blocks.append(f"[{src}]\n{summary.strip()}")
    if not blocks:
        return f"No knowledge-base matches for: {query!r}"
    return "\n\n---\n\n".join(blocks)


@tool
def rag_search_summarized(query: str) -> str:
    """Search the internal PDF knowledge base. Each matching PAPER is read in
    full and summarized with respect to your query by a helper model, so you
    get whole-paper insight (methods, circuits, results) instead of short
    excerpts. Slower than rag_search but much higher signal; make the query
    specific about what you want to learn."""
    return search_and_summarize_papers(query)


@tool
def web_search(query: str, max_searches: int = 5) -> str:
    """Search the web for current or external information."""
    tavily = TavilySearch(max_results= max_searches)
    return str(tavily.invoke({"query": query}))


tools = [rag_search, web_search, qiskit_docs]


# --------------------------------------------------------------------------- #
# LangGraph app  -- built lazily so importing this module needs no API keys
# --------------------------------------------------------------------------- #
def build_app():
    llm = init_chat_model(
        os.getenv("MODEL", "openai:gpt-4o"), temperature=0
    ).bind_tools(tools)

    def agent(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


if __name__ == "__main__":
    app = build_app()
    result = app.invoke({"messages": [("user", "your question here")]})
    print(result["messages"][-1].content)
