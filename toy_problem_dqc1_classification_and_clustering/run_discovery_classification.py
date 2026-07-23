"""
Run the generic AgentPipeline on the concrete QML feature-map DiscoveryTask

    # Pick any provider LangChain supports; set the matching API key in .env.
    export ANTHROPIC_API_KEY=...            # (or OPENAI_API_KEY, GOOGLE_API_KEY, ...)
    export AGENT_MODEL=claude-opus-4-8      # any init_chat_model id
    export AGENT_MODEL_PROVIDER=anthropic   # optional; inferred from the id if omitted
    export AGENT_DIM=4
    export AGENT_KERNEL=fidelity
    python run_discovery_classification.py

Optional:
    QML_MINIMIZE_RESOURCES=1   also ask the agent to minimize qubits/gates/depth
                               (secondary to accuracy)
    AGENT_EXPLORER_MODEL=...   cheap model for explorer/review/deploy to cut
    AGENT_REVIEW_MODEL=...     cost (each defaults to AGENT_MODEL); e.g.
    AGENT_DEPLOY_MODEL=...     claude-haiku-4-5-20251001

Swap make_qml_task(...) for any other DiscoveryTask to discover something else --
the pipeline itself does not change. Refresh seed baselines after changing
dim/kernel:  python classification_task.py baselines
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

import _bootstrap  # noqa: F401  (adds parent dreu-agent/ dir to sys.path)
from AgentPipeline import AgentPipeline
from classification_task import (make_qml_task, classical_baselines, format_classical_report,
                      save_seed_comparison_chart, QML_DIM, QML_KERNEL)
# RAG / web-search / docs tools (db/agent.py): rag_search hits the local PDF
# knowledge base (Chroma + OpenAI embeddings), web_search is Tavily, qiskit_docs
# is the offline Qiskit API reference. Import is cheap (retrievers are lazy).
from db.agent import (rag_search, web_search, qiskit_docs,
                      rag_search_summarized, search_and_summarize_papers)

# AGENT_MODEL is provider-agnostic (init_chat_model). ANTHROPIC_MODEL kept as a
# fallback for backward compatibility with older .env files.
MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
# Provider is optional: init_chat_model infers it from the model id (e.g. a
# "claude-*" id -> "anthropic", "gpt-*" -> "openai"). Set it to disambiguate.
PROVIDER = os.environ.get("AGENT_MODEL_PROVIDER") or None
# Per-role model overrides -- each defaults to AGENT_MODEL. Point the cheap
# roles at a small model (e.g. AGENT_EXPLORER_MODEL=claude-haiku-4-5-20251001)
# to cut cost; generate keeps AGENT_MODEL since it does the real design work.
EXPLORER_MODEL = os.environ.get("AGENT_EXPLORER_MODEL") or MODEL
REVIEW_MODEL = os.environ.get("AGENT_REVIEW_MODEL") or MODEL
DEPLOY_MODEL = os.environ.get("AGENT_DEPLOY_MODEL") or MODEL
DIM = QML_DIM          # from AGENT_DIM (defined in classification_task, honors .env)
KERNEL = QML_KERNEL    # from AGENT_KERNEL
MAX_ITERS = int(os.environ.get("AGENT_MAX_ITERS", "6"))
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.4"))
TRANSLATE_SPEC = os.environ.get("AGENT_TRANSLATE", "")  # e.g. "Port to PennyLane."
USE_EXPLORER = os.environ.get("AGENT_USE_EXPLORER", "1") != "0"  # 0 -> skip explorer
DEPLOY_PROMPT = os.environ.get("AGENT_DEPLOY_PROMPT", "")  # extra deploy instructions
# per-session cap on non-evaluate (research) tool calls; AgentPipeline default 4
MAX_RESEARCH = int(os.environ.get("AGENT_MAX_RESEARCH", "4"))


def make_llm(temperature: float, max_tokens: int, model: str | None = None):
    """Build a chat model for the configured MODEL/PROVIDER, any LangChain provider."""
    return init_chat_model(
        model or MODEL, model_provider=PROVIDER,
        temperature=temperature, max_tokens=max_tokens,
    )


def build_pipeline() -> AgentPipeline:
    task = make_qml_task(dim=DIM, kernel=KERNEL)

    # ---- a domain-expert persona (LLM) the reviewer can consult ----
    @tool
    def expert_consult(topic: str) -> str:
        """Consult a domain expert persona for improvement ideas on `topic`."""
        expert = make_llm(temperature=0.5, max_tokens=512, model=REVIEW_MODEL)
        msg = expert.invoke(
            f"You are a QML expert. In 2 sentences, advise on: {topic}")
        return msg.content if isinstance(msg.content, str) else str(msg.content)

    # ---- grounded expert: the QUESTION is the retrieval query (astronaut
    # pattern) -- papers are fetched from the knowledge base, summarized with
    # respect to the question, and the expert must answer from them ----
    @tool
    def expert_consult_grounded(question: str) -> str:
        """Ask a QML domain expert a specific question. The question itself is
        used to search the internal paper knowledge base; the expert answers
        grounded in what those papers say (and tells you when they don't cover
        it). Prefer this over ungrounded advice whenever prior work might
        answer the question -- phrase it as a full, specific question."""
        references = search_and_summarize_papers(question)
        expert = make_llm(temperature=0.3, max_tokens=1024, model=REVIEW_MODEL)
        msg = expert.invoke(
            "You are a quantum machine learning expert specializing in "
            "quantum feature map design.\n"
            "Answer the question below using the reference information as "
            "your primary source, supplemented by your expertise. Be specific "
            "and actionable. If the references do not cover the question, "
            "say so explicitly before answering from general knowledge.\n\n"
            f"## Question\n{question}\n\n"
            f"## Reference information (paper summaries)\n{references}")
        return msg.content if isinstance(msg.content, str) else str(msg.content)

    # Per-node toolsets (AgentPipeline also auto-adds view_seed_library to the
    # explorer + generate, check_explorer_notes to generate, and the evaluate
    # tool to generate):
    #   explorer : web_search + rag_search_summarized (external research +
    #              whole-paper digests from the knowledge base)
    #   generate : rag_search_summarized + qiskit_docs (knowledge base + API)
    #   review   : rag_search (cheap chunk lookup) + expert_consult_grounded
    explorer_tools = [web_search, rag_search_summarized]
    generate_tools = [rag_search_summarized, qiskit_docs]
    review_tools = [rag_search, expert_consult_grounded]

    generate_llm = make_llm(temperature=TEMPERATURE, max_tokens=4096)  # AGENT_MODEL
    explorer_llm = make_llm(temperature=0.5, max_tokens=2048, model=EXPLORER_MODEL)
    review_llm = make_llm(temperature=0.3, max_tokens=1024, model=REVIEW_MODEL)
    deploy_llm = make_llm(temperature=0.2, max_tokens=1024, model=DEPLOY_MODEL)
    translate_llm = make_llm(temperature=0.2, max_tokens=2048) if TRANSLATE_SPEC else None

    # The `evaluate` tool is built by the pipeline (it wraps task.evaluate) and is
    # called by the generate LLM through the ToolNode -- no need to wire it here.
    return AgentPipeline(
        task,
        generate_llm=generate_llm,
        use_explorer=USE_EXPLORER,         # False removes the explorer from the graph
        explorer_llm=explorer_llm,
        review_llm=review_llm,
        deploy_llm=deploy_llm,
        translate_llm=translate_llm,
        explorer_tools=explorer_tools,     # research phase (entry node)
        generate_tools=generate_tools,     # designer can research mid-design too
        review_tools=review_tools,         # None to disable
        deploy_prompt=DEPLOY_PROMPT or None,  # optional user preprocessing/deploy notes
        max_iters=MAX_ITERS,
        target_metric="accuracy",
        target_value=1.0,
        entry_point="build_circuit",
        translate_spec=TRANSLATE_SPEC or None,
        evaluate_kwargs={"plot": True},   # save a figure per evaluation
        max_session_research=MAX_RESEARCH,
    )


def print_seed_baselines(pipeline: AgentPipeline, best: dict) -> None:
    """Evaluate every seed feature map on the SAME dataset/kernel and print a
    comparison against the agent's best circuit; save a comparison chart and a
    paste-ready SEED_BASELINE_STATS block into the run's plots/ folder."""
    task = pipeline.task

    def _row(name: str, m: dict) -> str:
        if not m.get("ok"):
            return f"  {name:34s} ERROR: {m.get('error')}"
        return (f"  {name:34s} accuracy={m.get('accuracy'):<6} "
                f"qubits={m.get('n_qubits')} gates={m.get('n_gates')} "
                f"depth={m.get('depth')}")

    def _stat(m: dict) -> dict:
        return {"accuracy": round(m["accuracy"], 4),
                "qubits": m.get("n_qubits"), "gates": m.get("n_gates"),
                "depth": m.get("depth")}

    lines, stats = [], {}
    for name, code in task.seeds().items():
        # plot=True: each baseline saves a kernel/prediction figure into the
        # run's plots/ folder, comparable to the agent's iterN figures
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
        note=f"dim={DIM}, kernel={KERNEL}")
    if chart:
        lines.append(f"  [chart] {chart}")
    lines.append("\n# paste over SEED_BASELINE_STATS in classification_task.py:")
    lines.append("SEED_BASELINE_STATS = {")
    for n, s in stats.items():
        if not n.startswith(">>"):
            lines.append(f'    "{n}": {{"accuracy": {s["accuracy"]}, '
                         f'"qubits": {s["qubits"]}, "gates": {s["gates"]}, '
                         f'"depth": {s["depth"]}}},')
    lines.append("}")
    report = "\n".join(lines)
    print(report)
    task.log("SEED BASELINES", report)   # keep it in the run transcript too


def print_classical_baselines(pipeline: AgentPipeline) -> None:
    """Score classical-kernel SVM references (RBF + linear) on the same data;
    print a table and drop a config.txt into the run's plots/ folder."""
    task = pipeline.task
    res = classical_baselines(dim=DIM,
                              plot_dir=os.path.join(task.run_dir, "plots"))
    report = format_classical_report(res)
    print(report)
    task.log("CLASSICAL BASELINES", report)


if __name__ == "__main__":
    pipeline = build_pipeline()
    result = pipeline.run()

    print("\n" + "=" * 72)
    best = result["best"]
    if best.get("code"):
        print(f"BEST ARTIFACT -- variant #{best.get('variant', '?')} of "
              f"{result['iterations']} official variants "
              f"({result.get('trials', '?')} trials)")
        print("=" * 72)
        print(best["code"].strip())
        print("-" * 72)
        print("stats:", best.get("metrics"))
    else:
        print("No valid artifact was produced.")
    print("=" * 72)
    print("SUMMARY:\n", result["summary"])
    if result.get("translated"):
        print("\nTRANSLATED OUTPUT:\n", result["translated"])

    print("\n" + "=" * 72)
    print("SEED BASELINES (same dataset & kernel)")
    print("=" * 72)
    print_seed_baselines(pipeline, best)

    print("\n" + "=" * 72)
    print("CLASSICAL BASELINES (RBF + linear SVM, same data)")
    print("=" * 72)
    print_classical_baselines(pipeline)

    print("\nTranscript:", result["log_path"])
