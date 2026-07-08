"""Smoke tests for the db tools.

Run directly:      python test_agent.py
Or with pytest:    pytest test_agent.py -s

- qiskit_docs tests run fully offline (no API key needed).
- rag_search  needs OPENAI_API_KEY + a populated ./chroma_db (run ingest.py first).
- web_search  needs TAVILY_API_KEY.
Tests whose prerequisites are missing are SKIPPED, not failed.
"""

import os

from dotenv import load_dotenv

load_dotenv()

import agent


class SkipTest(Exception):
    """Raised to skip a test when its prerequisites (API keys / data) are absent."""


# --------------------------------------------------------------------------- #
# Qiskit reference (offline)
# --------------------------------------------------------------------------- #
def test_qiskit_reference_exact_name():
    out = agent.search_qiskit_reference("ZZFeatureMap")
    assert isinstance(out, str)
    assert "ZZFeatureMap" in out
    assert "feature_dimension" in out          # signature came through
    print(out[:400])


def test_qiskit_reference_keyword():
    out = agent.search_qiskit_reference("density matrix", top_k=3)
    assert "DensityMatrix" in out


def test_qiskit_reference_no_match():
    out = agent.search_qiskit_reference("thisisnotarealqiskitthing")
    assert "No Qiskit reference entries matched" in out


def test_qiskit_docs_tool():
    # Exercise it the way the LangGraph agent would call the tool.
    out = agent.qiskit_docs.invoke({"query": "PauliFeatureMap"})
    assert "PauliFeatureMap" in out
    print(out[:300])


# --------------------------------------------------------------------------- #
# RAG over PDFs (needs OPENAI_API_KEY + populated chroma_db)
# --------------------------------------------------------------------------- #
def test_rag_search():
    if not os.getenv("OPENAI_API_KEY"):
        raise SkipTest("OPENAI_API_KEY not set")
    out = agent.rag_search.invoke({"query": "quantum kernel feature map"})
    assert isinstance(out, str)
    print(out[:400] if out.strip() else "(knowledge base returned no chunks -- run ingest.py?)")


# --------------------------------------------------------------------------- #
# Web search (needs TAVILY_API_KEY)
# --------------------------------------------------------------------------- #
def test_web_search():
    if not os.getenv("TAVILY_API_KEY"):
        raise SkipTest("TAVILY_API_KEY not set")
    out = agent.web_search.invoke({"query": "DQC1 deterministic quantum computation one qubit"})
    assert isinstance(out, str) and len(out) > 0
    print(out[:400])


if __name__ == "__main__":
    tests = [
        test_qiskit_reference_exact_name,
        test_qiskit_reference_keyword,
        test_qiskit_reference_no_match,
        test_qiskit_docs_tool,
        test_rag_search,
        test_web_search,
    ]
    passed = failed = skipped = 0
    for t in tests:
        print(f"\n=== {t.__name__} ===")
        try:
            t()
        except SkipTest as e:
            print(f"SKIP: {e}")
            skipped += 1
        except Exception:
            import traceback

            traceback.print_exc()
            print("FAIL")
            failed += 1
        else:
            print("PASS")
            passed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
