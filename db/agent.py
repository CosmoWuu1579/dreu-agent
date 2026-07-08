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
