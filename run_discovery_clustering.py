"""
Run the generic AgentPipeline on the DQC1 entropy-clustering DiscoveryTask
(mirrors run_discovery.py, but the artifact is the feature map inside the
DQC1 normalized-trace kernel of dqc1.py and the objective is mean ARI).

    # Pick any provider LangChain supports; set the matching API key in .env.
    export ANTHROPIC_API_KEY=...            # (or OPENAI_API_KEY, GOOGLE_API_KEY, ...)
    export AGENT_MODEL=claude-opus-4-8      # any init_chat_model id
    export CLUSTER_DATASETS=spirals         # "spirals+moons+circles", ...
    export CLUSTER_N_POINTS=60              # kernel is O(N^2); keep small
    export CLUSTER_VARIANT=pre              # "pre" (DQC1-pre) or "full" (DQC1-full)
    python run_discovery_clustering.py

dqc1.py's own knobs (INIT_CLUSTERS, N_INIT, DATA_NOISE, DQC1_COMP_RANK, ...)
still control the clustering pipeline itself.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

from AgentPipeline import AgentPipeline
from clustering_task import (make_clustering_task, CLUSTER_DATASETS,
                             CLUSTER_N_POINTS, CLUSTER_VARIANT)

MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
PROVIDER = os.environ.get("AGENT_MODEL_PROVIDER") or None
MAX_ITERS = int(os.environ.get("AGENT_MAX_ITERS", "6"))
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.4"))
TRANSLATE_SPEC = os.environ.get("AGENT_TRANSLATE", "")
USE_EXPLORER = os.environ.get("AGENT_USE_EXPLORER", "1") != "0"
DEPLOY_PROMPT = os.environ.get("AGENT_DEPLOY_PROMPT", "")


def make_llm(temperature: float, max_tokens: int):
    return init_chat_model(
        MODEL, model_provider=PROVIDER,
        temperature=temperature, max_tokens=max_tokens,
    )


def build_pipeline() -> AgentPipeline:
    task = make_clustering_task(datasets=CLUSTER_DATASETS,
                                n_points=CLUSTER_N_POINTS,
                                variant=CLUSTER_VARIANT)

    # ---- optional research tools (same placeholders as run_discovery.py) ----
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
            "You are an expert in quantum kernels and entropy-based clustering "
            f"(DQC1, Renyi-2). In 2 sentences, advise on: {topic}")
        return msg.content if isinstance(msg.content, str) else str(msg.content)

    research_tools = [documentation_lookup, past_papers_lookup]

    generate_llm = make_llm(temperature=TEMPERATURE, max_tokens=4096)
    explorer_llm = make_llm(temperature=0.5, max_tokens=2048)
    review_llm = make_llm(temperature=0.3, max_tokens=1024)
    deploy_llm = make_llm(temperature=0.2, max_tokens=1024)
    translate_llm = make_llm(temperature=0.2, max_tokens=2048) if TRANSLATE_SPEC else None

    return AgentPipeline(
        task,
        generate_llm=generate_llm,
        use_explorer=USE_EXPLORER,
        explorer_llm=explorer_llm,
        review_llm=review_llm,
        deploy_llm=deploy_llm,
        translate_llm=translate_llm,
        explorer_tools=research_tools,
        generate_tools=research_tools,
        review_tools=[expert_consult],
        deploy_prompt=DEPLOY_PROMPT or None,
        max_iters=MAX_ITERS,
        target_metric="ari",               # mean ARI over CLUSTER_DATASETS
        target_value=1.0,
        entry_point="build_circuit",
        translate_spec=TRANSLATE_SPEC or None,
        evaluate_kwargs={"plot": True},    # save a labels figure per evaluation
        # each evaluation runs the full DQC1 clustering pipeline (slow), so
        # keep batches small
        max_variants_per_turn=2,
    )


if __name__ == "__main__":
    pipeline = build_pipeline()
    result = pipeline.run()

    print("\n" + "=" * 72)
    best = result["best"]
    if best.get("code"):
        print("BEST FEATURE MAP")
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
    print("\nTranscript:", result["log_path"])
