"""
Concrete DiscoveryTask for quantum feature-map design (mirrors the role of
eval_harness.py, but wraps it as a reusable DiscoveryTask). It plugs the existing
fidelity/DQC1 evaluator, seed library, and prompts into the generic framework.

Build one with make_qml_task(dim=..., kernel=...) and hand it to an AgentPipeline.
"""

from __future__ import annotations

from DiscoveryTask import DiscoveryTask
from eval_harness import evaluate_feature_map, load_dataset, SEED_FEATURE_MAPS


# --- prompts, one per agent role ------------------------------------------
def _system_prompts(dim: int) -> dict[str, str]:
    return {
        "explore": (
            f"You are a research assistant for designing quantum feature maps for a "
            f"{dim}-feature binary-classification quantum-kernel SVM. Use the lookup "
            f"tools to gather relevant encoding strategies, entanglement patterns, and "
            f"prior results. Take concise, actionable notes the designer can build on. "
            f"When you have enough, stop calling tools and summarize your notes."
        ),
        "generate": (
            f"You are an expert quantum circuit designer. Design a quantum feature "
            f"map for a {dim}-feature binary classification task that MAXIMIZES SVM "
            f"test accuracy while keeping the gate count modest.\n"
            f"Contract for the code you return:\n"
            f"- Define exactly one function build_circuit(x) -> QuantumCircuit.\n"
            f"- x is a length-{dim} numpy array. Use at least {dim} qubits, at most {dim + 4}.\n"
            f"- NO imports; `np` and `QuantumCircuit` are in scope. No measurements, "
            f"no trainable parameters, no classical nonlinear preprocessing of x.\n"
            f"- Present each candidate as a ```python code block, THEN call the "
            f"`evaluate` tool with that exact code to score it. To compare multiple "
            f"architectures in one round, call `evaluate` several times (once per "
            f"variant). After you see the scores, refine and evaluate again, or reply "
            f"WITHOUT a tool call when you want reviewer feedback. You may use the "
            f"lookup tools first if you need more information."
        ),
        "review": (
            "You are a critical reviewer of quantum feature-map designs for a "
            "quantum-kernel SVM. Given a candidate and its measured accuracy/gate "
            "stats, give ONE specific, actionable improvement (entanglement pattern, "
            "data re-uploading, encoding rotations, or lower depth). Two sentences max. "
            "If it errored, state the concrete fix."
        ),
        "deploy": (
            "You are writing the final report for an automated quantum-circuit search. "
            "Given the best feature map and its stats, explain in a short paragraph "
            "what was designed, its final accuracy/gate stats, and the design rationale."
        ),
    }


def make_qml_task(dim: int = 2, kernel: str = "fidelity",
                  train_size: int = 20, test_size: int = 5) -> DiscoveryTask:
    data = load_dataset(dim)  # (Xtr, ytr, Xte, yte); train/test sizes fixed in loader

    def _evaluate(code: str, *args, **kwargs) -> dict:
        # extra *args/**kwargs (e.g. plot=True, iteration=n) flow straight through
        return evaluate_feature_map(code, data=data, kernel=kernel, *args, **kwargs)

    return DiscoveryTask(
        name=f"qml_feature_map_dim{dim}_{kernel}",
        system_prompts=_system_prompts(dim),
        seeds={"feature_map": SEED_FEATURE_MAPS},
        evaluate_fn=_evaluate,
        # placeholders -- wire real RAG here later (see AgentPipeline plan)
        knowledge_sources=["arXiv:2210.09275", "arXiv:1804.11326 (Havlicek)"],
        documentation_source="qiskit (QuantumCircuit API)",
    )


if __name__ == "__main__":
    task = make_qml_task(dim=2)
    print(task.describe())
    print("seed types:", task.seed_types())
    seed = next(iter(task.seeds("feature_map").values()))
    print("baseline eval:", task.evaluate(seed))
