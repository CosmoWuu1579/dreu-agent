"""
Run the generic AgentPipeline on the concrete QML feature-map DiscoveryTask
(mirrors the role of agent_graph.py, but built on the reusable classes).

    # Pick any provider LangChain supports; set the matching API key in .env.
    export ANTHROPIC_API_KEY=...            # (or OPENAI_API_KEY, GOOGLE_API_KEY, ...)
    export AGENT_MODEL=claude-opus-4-8      # any init_chat_model id
    export AGENT_MODEL_PROVIDER=anthropic   # optional; inferred from the id if omitted
    export AGENT_DIM=4
    export AGENT_KERNEL=fidelity
    python run_discovery.py

Swap make_qml_task(...) for any other DiscoveryTask to discover something else --
the pipeline itself does not change.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

from AgentPipeline import AgentPipeline
from qml_task import make_qml_task

# AGENT_MODEL is provider-agnostic (init_chat_model). ANTHROPIC_MODEL kept as a
# fallback for backward compatibility with older .env files.
MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
# Provider is optional: init_chat_model infers it from the model id (e.g. a
# "claude-*" id -> "anthropic", "gpt-*" -> "openai"). Set it to disambiguate.
PROVIDER = os.environ.get("AGENT_MODEL_PROVIDER") or None
DIM = int(os.environ.get("AGENT_DIM", "2"))
KERNEL = os.environ.get("AGENT_KERNEL", "fidelity")
MAX_ITERS = int(os.environ.get("AGENT_MAX_ITERS", "6"))
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.4"))
TRANSLATE_SPEC = os.environ.get("AGENT_TRANSLATE", "")  # e.g. "Port to PennyLane."
USE_EXPLORER = os.environ.get("AGENT_USE_EXPLORER", "1") != "0"  # 0 -> skip explorer
DEPLOY_PROMPT = os.environ.get("AGENT_DEPLOY_PROMPT", "")  # extra deploy instructions


def make_llm(temperature: float, max_tokens: int):
    """Build a chat model for the configured MODEL/PROVIDER, any LangChain provider."""
    return init_chat_model(
        MODEL, model_provider=PROVIDER,
        temperature=temperature, max_tokens=max_tokens,
    )


def build_pipeline() -> AgentPipeline:
    task = make_qml_task(dim=DIM, kernel=KERNEL)

    # ---- optional tools the nodes may call (any can be omitted) ----
    # These wrap the task's RAG hooks; today they return placeholders, but the
    # wiring is real -- drop in a vector store behind task.retrieve*/ and they work.
    @tool
    def documentation_lookup(query: str) -> str:
        """Look up quantum-library API documentation relevant to `query`."""
        hits = task.retrieve_docs(query, k=3)
        return "\n".join(hits) if hits else f"[no docs indexed yet] source: {task.documentation_source()}"

    @tool
    def past_papers_lookup(query: str) -> str:
        """Retrieve relevant prior research / papers for `query`."""
        hits = task.retrieve(query, k=3)
        return "\n".join(hits) if hits else f"[no papers indexed yet] sources: {task.knowledge_sources()}"

    @tool
    def expert_consult(topic: str) -> str:
        """Consult a domain expert persona for improvement ideas on `topic`."""
        expert = make_llm(temperature=0.5, max_tokens=512)
        msg = expert.invoke(
            f"You are a QML expert. In 2 sentences, advise on: {topic}")
        return msg.content if isinstance(msg.content, str) else str(msg.content)

    # research_tools = [documentation_lookup, past_papers_lookup]
    research_tools = []
    generate_llm = make_llm(temperature=TEMPERATURE, max_tokens=4096)
    explorer_llm = make_llm(temperature=0.5, max_tokens=2048)
    review_llm = make_llm(temperature=0.3, max_tokens=1024)
    deploy_llm = make_llm(temperature=0.2, max_tokens=1024)
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
        explorer_tools=research_tools,     # research phase (entry node)
        generate_tools=research_tools,     # designer can research mid-design too
        review_tools=[expert_consult],     # None to disable
        deploy_prompt=DEPLOY_PROMPT or None,  # optional user preprocessing/deploy notes
        max_iters=MAX_ITERS,
        target_metric="accuracy",
        target_value=1.0,
        entry_point="build_circuit",
        translate_spec=TRANSLATE_SPEC or None,
        evaluate_kwargs={"plot": True},   # save a figure per evaluation
    )


def print_seed_baselines(pipeline: AgentPipeline, best: dict) -> None:
    """Evaluate every seed on the SAME dataset/kernel and print a comparison."""
    task = pipeline.task
    lines = []
    for name, code in task.seeds().items():
        m = task.evaluate(code)   # no plot=True: no figure spam for baselines
        if m.get("ok"):
            lines.append(f"  {name:26s} accuracy={m.get('accuracy'):<8} "
                         f"gates={m.get('n_gates')} depth={m.get('depth')}")
        else:
            lines.append(f"  {name:26s} ERROR: {m.get('error')}")
    bm = best.get("metrics") or {}
    if best.get("code") and bm.get("ok"):
        tag = f">> agent best (variant #{best.get('variant', '?')})"
        lines.append(f"  {tag:26s} accuracy={bm.get('accuracy'):<8} "
                     f"gates={bm.get('n_gates')} depth={bm.get('depth')}")
    report = "\n".join(lines)
    print(report)
    task.log("SEED BASELINES", report)   # keep it in the run transcript too


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

    print("\nTranscript:", result["log_path"])
