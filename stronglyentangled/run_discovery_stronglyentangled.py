"""
Run the generic AgentPipeline on the Strongly-Entangled quantum-head
DiscoveryTask (mirrors run_discovery.py / run_discovery_clustering.py, but built
for the trainable-VQC task in stronglyentangled_task.py).

    # Pick any provider LangChain supports; set the matching API key in .env.
    export ANTHROPIC_API_KEY=...            # (or OPENAI_API_KEY, GOOGLE_API_KEY, ...)
    export AGENT_MODEL=claude-opus-4-8      # any init_chat_model id
    export AGENT_DIM=2                      # features / qubits the head uses
    python run_discovery_stronglyentangled.py

Optional:
    SE_MINIMIZE_RESOURCES=1    also ask the agent to minimize qubits/gates/weights
    AGENT_MAX_ITERS=6          official design rounds (default 6)
    AGENT_USE_EXPLORER=0       skip the research/explorer phase
    AGENT_EXPLORER_MODEL / AGENT_REVIEW_MODEL / AGENT_DEPLOY_MODEL   cheap per-role models

The RAG / web-search / qiskit-docs research tools (db/agent.py) are OPTIONAL: if
that module (or its API keys) is unavailable, this runner falls back to no
research tools and an ungrounded expert consult, so it still runs end to end.
Refresh seed baselines after changing dim/data:
    python stronglyentangled_task.py baselines
"""

from __future__ import annotations

import os
import sys

# make the parent qml/ package importable (AgentPipeline etc. live one level up)
_QML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _QML_DIR not in sys.path:
    sys.path.insert(0, _QML_DIR)

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

from AgentPipeline import AgentPipeline
from stronglyentangled_task import (make_stronglyentangled_task, classical_baselines,
                                     format_classical_report,
                                     save_seed_comparison_chart, SE_DIM)

# Research tools are optional -- they need db/agent.py plus embedding/search API
# keys. Guard the import so the pipeline runs without them.
try:
    from db.agent import (rag_search, web_search, qiskit_docs,
                          rag_search_summarized, search_and_summarize_papers)
    HAVE_RESEARCH_TOOLS = True
except Exception as exc:                       # missing module or missing keys
    HAVE_RESEARCH_TOOLS = False
    print(f"[run_discovery_stronglyentangled] research tools unavailable "
          f"({type(exc).__name__}: {exc}); running without RAG/web search.")

MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
PROVIDER = os.environ.get("AGENT_MODEL_PROVIDER") or None
EXPLORER_MODEL = os.environ.get("AGENT_EXPLORER_MODEL") or MODEL
REVIEW_MODEL = os.environ.get("AGENT_REVIEW_MODEL") or MODEL
DEPLOY_MODEL = os.environ.get("AGENT_DEPLOY_MODEL") or MODEL
DIM = SE_DIM
MAX_ITERS = int(os.environ.get("AGENT_MAX_ITERS", "6"))
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.4"))
# explorer is only useful with research tools; default off when they are absent
USE_EXPLORER = (os.environ.get("AGENT_USE_EXPLORER", "1") != "0") and HAVE_RESEARCH_TOOLS
DEPLOY_PROMPT = os.environ.get("AGENT_DEPLOY_PROMPT", "")
MAX_RESEARCH = int(os.environ.get("AGENT_MAX_RESEARCH", "4"))


def make_llm(temperature: float, max_tokens: int, model: str | None = None):
    """Build a chat model for the configured MODEL/PROVIDER (any LangChain provider)."""
    return init_chat_model(
        model or MODEL, model_provider=PROVIDER,
        temperature=temperature, max_tokens=max_tokens,
    )


def build_pipeline() -> AgentPipeline:
    task = make_stronglyentangled_task(dim=DIM)

    # A domain-expert persona the reviewer can consult. When the paper knowledge
    # base is available we ground it in retrieved papers; otherwise it answers
    # from the model's own knowledge.
    if HAVE_RESEARCH_TOOLS:
        @tool
        def expert_consult_grounded(question: str) -> str:
            """Ask a QML expert a specific question, answered grounded in the
            internal paper knowledge base (say so when it doesn't cover it)."""
            references = search_and_summarize_papers(question)
            expert = make_llm(temperature=0.3, max_tokens=1024, model=REVIEW_MODEL)
            msg = expert.invoke(
                "You are a quantum machine learning expert specializing in "
                "variational quantum classifiers and entanglement design.\n"
                "Answer using the reference information as your primary source, "
                "supplemented by your expertise. If the references do not cover "
                "the question, say so before answering from general knowledge.\n\n"
                f"## Question\n{question}\n\n"
                f"## Reference information (paper summaries)\n{references}")
            return msg.content if isinstance(msg.content, str) else str(msg.content)

        expert_tool = expert_consult_grounded
        explorer_tools = [web_search, rag_search_summarized]
        generate_tools = [rag_search_summarized, qiskit_docs]
        review_tools = [rag_search, expert_consult_grounded]
    else:
        @tool
        def expert_consult(topic: str) -> str:
            """Consult a QML expert persona for ideas on `topic` (2 sentences)."""
            expert = make_llm(temperature=0.5, max_tokens=512, model=REVIEW_MODEL)
            msg = expert.invoke(
                f"You are a QML expert on entanglement design. In 2 sentences, "
                f"advise on: {topic}")
            return msg.content if isinstance(msg.content, str) else str(msg.content)

        expert_tool = expert_consult
        explorer_tools = []
        generate_tools = []
        review_tools = [expert_consult]

    generate_llm = make_llm(temperature=TEMPERATURE, max_tokens=4096)
    explorer_llm = make_llm(temperature=0.5, max_tokens=2048, model=EXPLORER_MODEL)
    review_llm = make_llm(temperature=0.3, max_tokens=1024, model=REVIEW_MODEL)
    deploy_llm = make_llm(temperature=0.2, max_tokens=1024, model=DEPLOY_MODEL)

    return AgentPipeline(
        task,
        generate_llm=generate_llm,
        use_explorer=USE_EXPLORER,
        explorer_llm=explorer_llm,
        review_llm=review_llm,
        deploy_llm=deploy_llm,
        explorer_tools=explorer_tools,
        generate_tools=generate_tools,
        review_tools=review_tools,
        deploy_prompt=DEPLOY_PROMPT or None,
        max_iters=MAX_ITERS,
        target_metric="accuracy",
        target_value=1.0,
        entry_point="create_qnn",
        evaluate_kwargs={"plot": True},   # save a figure per evaluation
        max_session_research=MAX_RESEARCH,
    )


def print_seed_baselines(pipeline: AgentPipeline, best: dict) -> None:
    """Score every seed head on the SAME data and compare to the agent's best;
    save a comparison chart into the run's plots/ folder."""
    task = pipeline.task

    def _row(name: str, m: dict) -> str:
        if not m.get("ok"):
            return f"  {name:34s} ERROR: {m.get('error')}"
        return (f"  {name:34s} accuracy={m.get('accuracy'):<6} "
                f"f1={m.get('f1')} qubits={m.get('n_qubits')} "
                f"gates={m.get('n_gates')} depth={m.get('depth')} "
                f"weights={m.get('n_weights')}")

    def _stat(m: dict) -> dict:
        return {"accuracy": round(m["accuracy"], 4), "f1": round(m["f1"], 4),
                "qubits": m.get("n_qubits"), "gates": m.get("n_gates"),
                "depth": m.get("depth"), "weights": m.get("n_weights")}

    lines, stats = [], {}
    for name, code in task.seeds().items():
        m = task.evaluate(code, plot=True, iteration=f"seed_{name}")
        lines.append(_row(name, m))
        if m.get("ok"):
            stats[name] = _stat(m)
    bm = best.get("metrics") or {}
    if best.get("code") and bm.get("ok"):
        tag = f">> agent best (variant #{best.get('variant', '?')})"
        lines.append(_row(tag, bm))
        stats[tag] = _stat(bm)
    chart = save_seed_comparison_chart(
        stats, os.path.join(task.run_dir, "plots", "seed_comparison.png"),
        note=f"dim={DIM}")
    if chart:
        lines.append(f"  [chart] {chart}")
    report = "\n".join(lines)
    print(report)
    task.log("SEED BASELINES", report)


def print_classical_baselines(pipeline: AgentPipeline) -> None:
    task = pipeline.task
    report = format_classical_report(classical_baselines(dim=DIM))
    print(report)
    task.log("CLASSICAL BASELINES", report)


if __name__ == "__main__":
    pipeline = build_pipeline()
    result = pipeline.run()

    print("\n" + "=" * 72)
    best = result["best"]
    if best.get("code"):
        print(f"BEST QUANTUM HEAD -- variant #{best.get('variant', '?')} of "
              f"{result['iterations']} official variants "
              f"({result.get('trials', '?')} trials)")
        print("=" * 72)
        print(best["code"].strip())
        print("-" * 72)
        print("stats:", best.get("metrics"))
    else:
        print("No valid quantum head was produced.")
    print("=" * 72)
    print("SUMMARY:\n", result["summary"])

    print("\n" + "=" * 72)
    print("SEED BASELINES (same features)")
    print("=" * 72)
    print_seed_baselines(pipeline, best)

    print("\n" + "=" * 72)
    print("CLASSICAL BASELINES (RBF + linear SVM, same features)")
    print("=" * 72)
    print_classical_baselines(pipeline)

    print("\nTranscript:", result["log_path"])
