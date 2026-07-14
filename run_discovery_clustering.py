"""
Run the generic AgentPipeline on the DQC1 entropy-clustering DiscoveryTask
(mirrors run_discovery.py, but the artifact is the feature map inside the
DQC1 normalized-trace kernel of dqc1.py and the objective is mean ARI).

    # Pick any provider LangChain supports; set the matching API key in .env.
    export ANTHROPIC_API_KEY=...            # (or OPENAI_API_KEY, GOOGLE_API_KEY, ...)
    export AGENT_MODEL=claude-opus-4-8      # any init_chat_model id
    export CLUSTER_DATASETS=spirals         # "spirals+moons+circles", ...
    export CLUSTER_N_POINTS=100             # kernel is O(N^2); keep modest
    export CLUSTER_VARIANT=full             # "full" (DQC1-full; default) or "pre"
                                            # NOTE: "pre" can be feature-map
                                            # INSENSITIVE at small N (see
                                            # clustering_task.py docstring)
    python run_discovery_clustering.py

dqc1.py's own knobs (INIT_CLUSTERS, N_INIT, DATA_NOISE, DQC1_COMP_RANK, ...)
still control the clustering pipeline itself. Optional:
    CLUSTER_MINIMIZE_RESOURCES=1   also ask the agent to minimize qubits/
                                   gates/depth (secondary to ARI)
    AGENT_EXPLORER_MODEL=...       cheap model for explorer/review/deploy to
    AGENT_REVIEW_MODEL=...         cut cost (each defaults to AGENT_MODEL);
    AGENT_DEPLOY_MODEL=...         e.g. claude-haiku-4-5-20251001
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

from AgentPipeline import AgentPipeline
from clustering_task import (make_clustering_task, classical_baselines,
                             format_classical_report,
                             save_seed_comparison_chart, CLUSTER_DATASETS,
                             CLUSTER_N_POINTS, CLUSTER_VARIANT)
# RAG / web-search / docs tools (db/agent.py): rag_search hits the local PDF
# knowledge base (Chroma + OpenAI embeddings), web_search is Tavily, qiskit_docs
# is the offline Qiskit API reference. Import is cheap (retrievers are lazy).
from db.agent import (rag_search, web_search, qiskit_docs,
                      rag_search_summarized, search_and_summarize_papers)

MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
PROVIDER = os.environ.get("AGENT_MODEL_PROVIDER") or None
# Per-role model overrides -- each defaults to AGENT_MODEL. Point the cheap
# roles at a small model (e.g. AGENT_EXPLORER_MODEL=claude-haiku-4-5-20251001)
# to cut cost; generate keeps AGENT_MODEL since it does the real design work.
EXPLORER_MODEL = os.environ.get("AGENT_EXPLORER_MODEL") or MODEL
REVIEW_MODEL = os.environ.get("AGENT_REVIEW_MODEL") or MODEL
DEPLOY_MODEL = os.environ.get("AGENT_DEPLOY_MODEL") or MODEL
MAX_ITERS = int(os.environ.get("AGENT_MAX_ITERS", "6"))
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.4"))
TRANSLATE_SPEC = os.environ.get("AGENT_TRANSLATE", "")
USE_EXPLORER = os.environ.get("AGENT_USE_EXPLORER", "1") != "0"
DEPLOY_PROMPT = os.environ.get("AGENT_DEPLOY_PROMPT", "")
# per-session cap on non-evaluate (research) tool calls; AgentPipeline default 4
MAX_RESEARCH = int(os.environ.get("AGENT_MAX_RESEARCH", "4"))


def make_llm(temperature: float, max_tokens: int, model: str | None = None):
    return init_chat_model(
        model or MODEL, model_provider=PROVIDER,
        temperature=temperature, max_tokens=max_tokens,
    )


def build_pipeline() -> AgentPipeline:
    task = make_clustering_task(datasets=CLUSTER_DATASETS,
                                n_points=CLUSTER_N_POINTS,
                                variant=CLUSTER_VARIANT)

    # ---- a domain-expert persona (LLM) the reviewer can consult ----
    @tool
    def expert_consult(topic: str) -> str:
        """Consult a domain expert persona for improvement ideas on `topic`."""
        expert = make_llm(temperature=0.5, max_tokens=512, model=REVIEW_MODEL)
        msg = expert.invoke(
            "You are an expert in quantum kernels and entropy-based clustering "
            f"(DQC1, Renyi-2). In 2 sentences, advise on: {topic}")
        return msg.content if isinstance(msg.content, str) else str(msg.content)

    # ---- grounded expert: the QUESTION is the retrieval query (astronaut
    # pattern) -- papers are fetched from the knowledge base, summarized with
    # respect to the question, and the expert must answer from them ----
    @tool
    def expert_consult_grounded(question: str) -> str:
        """Ask a domain expert a specific question. The question itself is
        used to search the internal paper knowledge base; the expert answers
        grounded in what those papers say (and tells you when they don't cover
        it). Prefer this over ungrounded advice whenever prior work might
        answer the question -- phrase it as a full, specific question."""
        references = search_and_summarize_papers(question)
        expert = make_llm(temperature=0.3, max_tokens=1024, model=REVIEW_MODEL)
        msg = expert.invoke(
            "You are an expert in quantum kernels and entropy-based "
            "clustering (DQC1, Renyi-2), specializing in quantum feature map "
            "design for trace kernels.\n"
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

    return AgentPipeline(
        task,
        generate_llm=generate_llm,
        use_explorer=USE_EXPLORER,
        explorer_llm=explorer_llm,
        review_llm=review_llm,
        deploy_llm=deploy_llm,
        translate_llm=translate_llm,
        explorer_tools=explorer_tools,
        generate_tools=generate_tools,
        review_tools=review_tools,
        deploy_prompt=DEPLOY_PROMPT or None,
        max_iters=MAX_ITERS,
        target_metric="ari",               # mean ARI over CLUSTER_DATASETS
        target_value=1.0,
        entry_point="build_circuit",
        translate_spec=TRANSLATE_SPEC or None,
        evaluate_kwargs={"plot": True},    # save a labels figure per evaluation
        # each trial runs the full DQC1 clustering pipeline (~1 min at the
        # default 60 points), so keep the per-session scratch budget small
        max_session_trials=3,
        max_session_research=MAX_RESEARCH,
    )


def print_seed_baselines(pipeline: AgentPipeline, best: dict) -> None:
    """Evaluate every seed feature map on the SAME datasets/variant and print
    a comparison against the agent's best circuit (each row runs the full
    DQC1 clustering pipeline, so this takes ~1 min per seed)."""
    task = pipeline.task

    def _row(name: str, m: dict) -> str:
        if not m.get("ok"):
            return f"  {name:34s} ERROR: {m.get('error')}"
        per = "  ".join(
            f"{d}: ARI={s['ARI']:.3f} (kmeans {s['kmeans_ARI']:.3f})"
            for d, s in (m.get("per_dataset") or {}).items())
        return (f"  {name:34s} ari={m['ari']:.4f} nmi={m['nmi']:.4f} "
                f"qubits={m.get('n_qubits')} gates={m.get('n_gates')} "
                f"depth={m.get('depth')} | {per}")

    def _stat(m: dict) -> dict:
        return {"ari": round(m["ari"], 4), "nmi": round(m["nmi"], 4),
                "qubits": m.get("n_qubits"), "gates": m.get("n_gates"),
                "depth": m.get("depth")}

    lines, stats = [], {}
    for name, code in task.seeds().items():
        # plot=True: each baseline saves a labeled figure into the run's
        # plots/ folder (clustering_<stamp>_iterseed_<name>.png), so seed
        # clusterings can be compared visually against the agent's iterN ones
        m = task.evaluate(code, plot=True, iteration=f"seed_{name}")
        lines.append(_row(name, m))
        if m.get("ok"):
            stats[name] = _stat(m)
    bm = best.get("metrics") or {}
    if best.get("code") and bm.get("ok"):
        tag = f">> agent best (variant #{best.get('variant', '?')})"
        lines.append(_row(tag, bm))
        stats[tag] = _stat(bm)
    # bar chart of every seed + the agent, into the run's plots/ folder
    chart = save_seed_comparison_chart(
        stats, os.path.join(task.run_dir, "plots", "seed_comparison.png"),
        note=f"{CLUSTER_DATASETS}, DQC1-{CLUSTER_VARIANT}, N={CLUSTER_N_POINTS}")
    if chart:
        lines.append(f"  [chart] {chart}")
    # paste-ready refresh for clustering_task.SEED_BASELINE_STATS (seeds only)
    lines.append("\n# paste over SEED_BASELINE_STATS in clustering_task.py:")
    lines.append("SEED_BASELINE_STATS = {")
    for n, s in stats.items():
        if not n.startswith(">>"):
            lines.append(f'    "{n}": {{"ari": {s["ari"]}, "nmi": {s["nmi"]}, '
                         f'"qubits": {s["qubits"]}, "gates": {s["gates"]}, '
                         f'"depth": {s["depth"]}}},')
    lines.append("}")
    report = "\n".join(lines)
    print(report)
    task.log("SEED BASELINES", report)   # keep it in the run transcript too


def print_classical_baselines(pipeline: AgentPipeline) -> None:
    """Score dqc1.py's classical methods (Parzen FULL Gaussian + k-means) on
    the same datasets; prints a table and saves a panel figure into the run's
    plots/ folder."""
    task = pipeline.task
    res = classical_baselines(plot_dir=os.path.join(task.run_dir, "plots"))
    report = format_classical_report(res)
    print(report)
    task.log("CLASSICAL BASELINES", report)


if __name__ == "__main__":
    pipeline = build_pipeline()
    result = pipeline.run()

    print("\n" + "=" * 72)
    best = result["best"]
    if best.get("code"):
        print(f"BEST FEATURE MAP -- variant #{best.get('variant', '?')} of "
              f"{result['iterations']} official variants "
              f"({result.get('trials', '?')} trials)")
        print("=" * 72)
        print(best["code"].strip())
        print("-" * 72)
        print("stats:", best.get("metrics"))
    else:
        print("No valid feature map was produced.")
    print("=" * 72)
    print("SUMMARY:\n", result["summary"])
    if result.get("translated"):
        print("\nTRANSLATED OUTPUT:\n", result["translated"])

    print("\n" + "=" * 72)
    print("SEED BASELINES (same datasets, variant & pipeline settings)")
    print("=" * 72)
    print_seed_baselines(pipeline, best)

    print("\n" + "=" * 72)
    print("CLASSICAL BASELINES (Parzen FULL Gaussian + k-means, dqc1.py)")
    print("=" * 72)
    print_classical_baselines(pipeline)

    print("\nTranscript:", result["log_path"])
