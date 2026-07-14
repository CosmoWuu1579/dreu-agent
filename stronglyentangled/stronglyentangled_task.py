"""
Concrete DiscoveryTask for the "Strongly Entangled" hybrid-QCNN experiment
(mirrors the role of qml_task.py / clustering_task.py). The candidate artifact
is a TRAINABLE quantum head create_qnn(n) -> EstimatorQNN; the actual training /
scoring lives in stronglyentangled_eval.py (styled after the paper's source).

Paper: Cui & Huang, "Brain Tumor Detection: Strong Entanglement Improves
Quantum Neural Network's Classification Ability" (ICONIP 2024). A CNN compresses
each MRI to a few features; a small quantum head (encoder + trainable VQC)
consumes them; its expectation is the class score. Finding: MORE ENTANGLEMENT in
the encoder/VQC -> better classification, at a tiny parameter cost.

This task automates that experiment: the LLM designs the ENTANGLEMENT PATTERN of
the head, and stronglyentangled_eval.evaluate_qcnn trains + scores it on fixed
(frozen-CNN-proxy) features. See stronglyentangled_eval.py for the training loop.

Env knobs (data/training knobs are read in stronglyentangled_eval.py):
    AGENT_DIM         number of features / qubits the head uses (default 2)
    SE_TRAIN_SIZE / SE_TEST_SIZE / SE_EPOCHS / SE_LR / SE_DATA_SEED  (eval knobs)
    SE_MINIMIZE_RESOURCES  1 -> also push for fewer qubits/gates/weights/depth
    SE_MAX_QUBITS / SE_MAX_GATES / SE_MAX_DEPTH / SE_MAX_WEIGHTS  hard caps
                      (only enforced when SE_MINIMIZE_RESOURCES=1; 0/unset = off)

Refresh the seed baselines after changing dim/data:
    python stronglyentangled_task.py baselines   # per-seed figures + paste-ready stats
    python stronglyentangled_task.py classical    # classical-SVM references only
"""

from __future__ import annotations

import os
import sys
import datetime

# make the parent qml/ package importable (DiscoveryTask lives one level up)
_QML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _QML_DIR not in sys.path:
    sys.path.insert(0, _QML_DIR)

# Honor .env even when THIS module is the entry point (mirrors qml_task.py):
# load_dotenv does NOT override already-set vars, so it is a no-op when
# run_discovery_stronglyentangled.py already called it. Must precede env reads.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: save figures, never open a window
import matplotlib.pyplot as plt

from DiscoveryTask import DiscoveryTask
# evaluator + data + training config live in the eval module (styled after the
# paper's source), so the task file stays about prompts / seeds / baselines
from stronglyentangled_eval import (evaluate_qcnn, load_se_data,
                                     classical_baselines, format_classical_report,
                                     SE_DIM, SE_TRAIN_SIZE, SE_TEST_SIZE,
                                     SE_EPOCHS, SE_LR, SE_DATA_SEED)

SE_MINIMIZE_RESOURCES = os.environ.get("SE_MINIMIZE_RESOURCES", "0") != "0"
# SE_FULL=1 -> score candidates with the FAITHFUL end-to-end evaluator (trains
# the whole CNN+quantum net on Br35H, CNN unfrozen). Needs the dataset + a GPU
# and is MUCH slower. Default off: the fast frozen-feature proxy.
SE_FULL = os.environ.get("SE_FULL", "0") != "0"
# SE_FEATURES selects WHAT the frozen evaluator classifies (ignored when SE_FULL,
# since the full evaluator trains on raw images):
#   "adhoc" (default) -> qiskit ad_hoc stand-in vectors (fast, no dataset needed)
#   "br35h"           -> real 512-d frozen-CNN features from Br35H (needs dataset
#                        + torchvision; "the swap" -> see br35h_features.py)
SE_FEATURES = os.environ.get("SE_FEATURES", "adhoc").strip().lower()


def active_evaluator():
    """The evaluator to use given SE_FULL: the faithful end-to-end trainer when
    SE_FULL=1 (lazy import so torchvision/Br35H are only needed then), else the
    fast frozen-feature proxy. Used by BOTH the discovery loop (make_*_task) and
    the standalone `baselines` command, so they never disagree."""
    if SE_FULL:
        from stronglyentangled_eval_full import evaluate_qcnn_full
        return evaluate_qcnn_full
    return evaluate_qcnn


def _baseline_data(dim: int):
    """Dataset arg for the active (frozen) evaluator:
      SE_FULL          -> None (the full evaluator loads Br35H images itself)
      SE_FEATURES=br35h-> real frozen-CNN features from Br35H (lazy import so
                          torchvision/dataset are only required in this mode)
      otherwise        -> the qiskit ad_hoc stand-in features."""
    if SE_FULL:
        return None
    if SE_FEATURES == "br35h":
        from br35h_features import load_br35h_features
        return load_br35h_features()
    return load_se_data(dim)


def _limit(name: str) -> int | None:
    """A hard resource cap from the env: unset / "" / "0" -> None (no limit)."""
    v = os.environ.get(name, "").strip()
    return int(v) if v and v != "0" else None


# HARD budgets, ONLY enforced when SE_MINIMIZE_RESOURCES is on. An over-budget
# head is still trained/scored but marked ok=False so it cannot win.
SE_MAX_QUBITS = _limit("SE_MAX_QUBITS")
SE_MAX_GATES = _limit("SE_MAX_GATES")
SE_MAX_DEPTH = _limit("SE_MAX_DEPTH")
SE_MAX_WEIGHTS = _limit("SE_MAX_WEIGHTS")     # trainable QNN weights (paper's "#Para")


# ---------------------------------------------------------------------------
# Seeds: quantum heads spanning the ENTANGLEMENT axis the paper studies.
# Each defines create_qnn(n) -> EstimatorQNN under the agent's contract.
# E1 (ZFeatureMap) = product encoder, NO entanglement.  E2 (ZZFeatureMap) =
# pairwise-entangled encoder (the paper's stronger setting).
# ---------------------------------------------------------------------------
SEED_QNNS = {
    # --- Encoder 1: product encoder (NO encoder entanglement) -------------
    "qce1_z_realamp": """
def create_qnn(n):
    fmap = z_feature_map(n)                    # product encoder: no entanglement
    ansatz = real_amplitudes(n, reps=1)        # trainable VQC
    qc = QuantumCircuit(n)
    qc.compose(fmap, inplace=True)
    qc.compose(ansatz, inplace=True)
    obs = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1)])   # Z on last qubit
    return EstimatorQNN(circuit=qc, observables=obs,
                        input_params=fmap.parameters,
                        weight_params=ansatz.parameters,
                        input_gradients=True)
""",
    # --- Encoder 2: ZZ-entangled encoder (the paper's winning flavor) -----
    "qce2_zz_realamp": """
def create_qnn(n):
    fmap = zz_feature_map(n)                   # pairwise-entangled encoder
    ansatz = real_amplitudes(n, reps=1)
    qc = QuantumCircuit(n)
    qc.compose(fmap, inplace=True)
    qc.compose(ansatz, inplace=True)
    obs = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1)])
    return EstimatorQNN(circuit=qc, observables=obs,
                        input_params=fmap.parameters,
                        weight_params=ansatz.parameters,
                        input_gradients=True)
""",
    # --- E2 + a deeper, linearly-entangled trainable ansatz ---------------
    "qce2_zz_linear_reps2": """
def create_qnn(n):
    fmap = zz_feature_map(n, reps=2)
    ansatz = real_amplitudes(n, reps=2, entanglement="linear")
    qc = QuantumCircuit(n)
    qc.compose(fmap, inplace=True)
    qc.compose(ansatz, inplace=True)
    obs = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1)])
    return EstimatorQNN(circuit=qc, observables=obs,
                        input_params=fmap.parameters,
                        weight_params=ansatz.parameters,
                        input_gradients=True)
""",
    # --- E2 + fully-entangled efficient_su2 (RY+RZ layers, all-to-all CX) --
    "qce2_zz_efficientsu2_full": """
def create_qnn(n):
    fmap = zz_feature_map(n)
    ansatz = efficient_su2(n, reps=1, entanglement="full")
    qc = QuantumCircuit(n)
    qc.compose(fmap, inplace=True)
    qc.compose(ansatz, inplace=True)
    obs = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1)])
    return EstimatorQNN(circuit=qc, observables=obs,
                        input_params=fmap.parameters,
                        weight_params=ansatz.parameters,
                        input_gradients=True)
""",
    # --- Hand-built entangler: explicit CX chain you can restructure ------
    # (the clearest place to edit the ENTANGLEMENT PATTERN by hand)
    "custom_cx_chain": """
def create_qnn(n):
    fmap = zz_feature_map(n)
    t = ParameterVector("t", 3 * n)            # trainable weights
    ansatz = QuantumCircuit(n)
    k = 0
    for q in range(n):
        ansatz.ry(t[k], q); k += 1
    for q in range(n - 1):                      # linear CX entangler
        ansatz.cx(q, q + 1)
    for q in range(n):
        ansatz.ry(t[k], q); k += 1
    for q in range(0, n - 1, 2):                # second, offset entangler
        ansatz.cx(q, q + 1)
    for q in range(n):
        ansatz.rz(t[k], q); k += 1
    qc = QuantumCircuit(n)
    qc.compose(fmap, inplace=True)
    qc.compose(ansatz, inplace=True)
    obs = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1)])
    return EstimatorQNN(circuit=qc, observables=obs,
                        input_params=fmap.parameters, weight_params=list(t),
                        input_gradients=True)
""",
}


# ---------------------------------------------------------------------------
# Measured seed baselines: shown to the agent as "the numbers to beat".
# EMPTY by default -- run `python stronglyentangled_task.py baselines` and paste
# the printed dict here. Until then the agent simply has no baseline table.
# ---------------------------------------------------------------------------
SEED_BASELINE_CONFIG = (f"dim={SE_DIM}, ad_hoc/synthetic, train={SE_TRAIN_SIZE}, "
                        f"test={SE_TEST_SIZE}, epochs={SE_EPOCHS}, lr={SE_LR}, "
                        f"seed={SE_DATA_SEED}")
SEED_BASELINE_STATS: dict = {}


def _annotated_seeds(seed_stats: dict | None) -> dict:
    """Seed library with each head's measured accuracy prepended as a comment."""
    if not seed_stats:
        return dict(SEED_QNNS)
    out = {}
    for name, code in SEED_QNNS.items():
        s = seed_stats.get(name)
        if isinstance(s, dict) and "accuracy" in s:
            header = (f"# measured ({SEED_BASELINE_CONFIG}): "
                      f"accuracy={s['accuracy']:.3f}, f1={s.get('f1', '?')}, "
                      f"qubits={s.get('qubits', '?')}, gates={s['gates']}, "
                      f"depth={s['depth']}, weights={s.get('weights', '?')}")
        else:
            header = "# (no measured baseline yet)"
        out[name] = f"\n{header}\n{code.strip()}\n"
    return out


def save_seed_comparison_chart(stats: dict, path: str, note: str = "") -> str:
    """Horizontal bars of test accuracy per head, best on top; '>>' rows bold."""
    rows = [(n, s) for n, s in stats.items()
            if isinstance(s, dict) and "accuracy" in s]
    if not rows:
        return ""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows.sort(key=lambda kv: kv[1]["accuracy"])
    names = [n for n, _ in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 2))
    bars = ax.barh(y, [s["accuracy"] for _, s in rows], color="#2563eb")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_yticks(y, names, fontsize=9)
    for tick, name in zip(ax.get_yticklabels(), names):
        if name.startswith(">>"):
            tick.set_fontweight("bold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("test accuracy")
    ax.set_title("Quantum-head comparison" + (f" — {note}" if note else ""),
                 fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compute_seed_baselines(dim: int = SE_DIM,
                           plot_dir: str | None = "seed_baselines") -> dict:
    """Train+score every seed head with the real evaluator; return
    {name: {"accuracy","f1","qubits","gates","depth","weights"}} and print a
    paste-ready SEED_BASELINE_STATS block (+ a comparison chart with plot_dir).

    Honors SE_FULL: uses the SAME evaluator as the discovery loop (faithful
    end-to-end when SE_FULL=1, else the frozen proxy), so baselines and the
    agent's scores are always comparable."""
    evaluator = active_evaluator()
    data = _baseline_data(dim)
    stats: dict = {}
    for name, code in SEED_QNNS.items():
        m = evaluator(code, data=data, n=dim, plot=bool(plot_dir),
                      iteration=f"seed_{name}", plot_dir=plot_dir or "plots")
        if m.get("ok"):
            stats[name] = {"accuracy": round(m["accuracy"], 4),
                           "f1": round(m["f1"], 4), "qubits": m["n_qubits"],
                           "gates": m["n_gates"], "depth": m["depth"],
                           "weights": m["n_weights"]}
            print(f"  {name:28s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                  f"qubits={m['n_qubits']} gates={m['n_gates']} "
                  f"depth={m['depth']} weights={m['n_weights']}")
        else:
            stats[name] = {"error": m.get("error")}
            print(f"  {name:28s} ERROR: {m.get('error')}")
    if plot_dir:
        chart = save_seed_comparison_chart(
            stats, os.path.join(plot_dir, "seed_comparison.png"),
            note=SEED_BASELINE_CONFIG)
        print(f"  [chart] {chart}")
    print("\n# paste over SEED_BASELINE_STATS in stronglyentangled_task.py:")
    print(f'SEED_BASELINE_CONFIG = "{SEED_BASELINE_CONFIG}"')
    print("SEED_BASELINE_STATS = {")
    for n, s in stats.items():
        if "accuracy" in s:
            print(f'    "{n}": {{"accuracy": {s["accuracy"]}, "f1": {s["f1"]}, '
                  f'"qubits": {s["qubits"]}, "gates": {s["gates"]}, '
                  f'"depth": {s["depth"]}, "weights": {s["weights"]}}},')
    print("}")
    return stats


# ---------------------------------------------------------------------------
# Prompts, one per agent role
# ---------------------------------------------------------------------------
def _system_prompts(dim: int, seed_stats: dict | None = None,
                    minimize_resources: bool = False) -> dict[str, str]:
    q_hi = (SE_MAX_QUBITS if (minimize_resources and SE_MAX_QUBITS is not None)
            else dim + 4)
    resource_block, resource_note = "", ""
    if minimize_resources:
        limits = []
        if SE_MAX_QUBITS is not None:
            limits.append(f"at most {SE_MAX_QUBITS} qubits")
        if SE_MAX_GATES is not None:
            limits.append(f"at most {SE_MAX_GATES} gates")
        if SE_MAX_DEPTH is not None:
            limits.append(f"depth at most {SE_MAX_DEPTH}")
        if SE_MAX_WEIGHTS is not None:
            limits.append(f"at most {SE_MAX_WEIGHTS} trainable weights")
        limit_line = ""
        if limits:
            limit_line = (
                "\nHARD RESOURCE LIMITS (ENFORCED): your head must use "
                + ", ".join(limits) + ". A head that exceeds ANY of these is still "
                "trained/scored but marked NOT VIABLE (ok=false) no matter how "
                "accurate -- it cannot win, and the result says by how much it "
                "went over.")
        resource_block = (
            "\n\n# Secondary objective: circuit economy\n"
            "Among heads with comparable accuracy, PREFER fewer qubits, gates, "
            "trainable weights, and lower depth (all reported every evaluation). "
            "The paper's whole claim is BIG accuracy gains for a TINY parameter "
            "increase -- honor that: beat the baseline first, then trim any gate "
            "or weight that does not earn its accuracy." + limit_line)
        resource_note = (" Also flag waste: qubits, gates, weights, or depth "
                         "that do not earn their accuracy.")

    rows = [(n, s) for n, s in (seed_stats or {}).items()
            if isinstance(s, dict) and "accuracy" in s]
    beat_note, baseline_block = "", ""
    if rows:
        best_name, best_s = max(rows, key=lambda kv: kv[1]["accuracy"])
        table = "\n".join(
            f"  - {n}: accuracy={s['accuracy']:.3f}, f1={s.get('f1', '?')}, "
            f"qubits={s.get('qubits', '?')}, gates={s['gates']}, "
            f"depth={s['depth']}, weights={s.get('weights', '?')}"
            for n, s in rows)
        beat_note = (
            f" Your objective is to IMPROVE on the best head so far -- from the "
            f"start that is the strongest seed, {best_name} (accuracy "
            f"{best_s['accuracy']:.3f}, {best_s.get('weights', '?')} weights) -- by "
            f"matching or beating its accuracy while keeping the trainable-weight "
            f"count small.")
        baseline_block = (
            f"\n\n# Seed baselines to beat ({SEED_BASELINE_CONFIG})\n{table}\n"
            f"NUMBER TO BEAT: {best_name} at accuracy {best_s['accuracy']:.3f} "
            f"({best_s.get('weights', '?')} weights). Success = accuracy strictly "
            f"ABOVE it; a strong result also uses NO MORE trainable weights for the "
            f"same-or-better accuracy. First reason about WHY entanglement helps "
            f"(compare the E1 vs E2 seeds), then push that structure further.\n"
            f"SUBMISSION RULE: submit your OWN head, never a seed verbatim (nor a "
            f"cosmetic edit). Even if nothing beat {best_name}, submit your best "
            f"original and say it fell short -- returning a seed is a failure.")

    contract = (
        f"Contract: define exactly one create_qnn(n) -> EstimatorQNN. n={dim} is "
        f"the number of feature inputs (and the minimum qubit count); use "
        f"{dim}-{q_hi} qubits. Set input_params to the n encoder inputs, "
        f"weight_params to the trainable weights, input_gradients=True, and use "
        f"exactly ONE observable (a single Z, e.g. Z on the last qubit) so the "
        f"head outputs one score. NO imports -- these are already in scope: np, "
        f"QuantumCircuit, ParameterVector, z_feature_map, zz_feature_map, "
        f"real_amplitudes, efficient_su2, SparsePauliOp, EstimatorQNN. The head is "
        f"trained for you (NAdam + cross-entropy) inside a fixed fc2->QNN->fc3 "
        f"net on FIXED features, so do NOT add measurements or classical "
        f"preprocessing. Present each candidate as a ```python block, THEN call "
        f"`evaluate` on it. Submit by replying WITHOUT a tool call.")

    return {
        "explore": (
            f"You are a research assistant surveying prior work to inform the "
            f"design of a TRAINABLE quantum classification head (encoder + "
            f"variational circuit) for a {dim}-feature binary tumor-vs-healthy "
            f"task. The open question (from Cui & Huang 2024): does STRONGER "
            f"ENTANGLEMENT in the encoder/VQC improve accuracy at low parameter "
            f"cost? Use the lookup tools to cover, in priority order:\n"
            f"1. Background: quantum CNNs / variational classifiers, entanglement "
            f"and expressibility, barren plateaus / overfitting on small data.\n"
            f"2. Comparison: survey the seed heads AND ansatz/encoder families "
            f"from prior work (ZZ feature maps, RealAmplitudes, EfficientSU2, "
            f"conv-pool QCNN ansatze).\n"
            f"3. Directions (MOST important): concrete entanglement patterns worth "
            f"trying here and why.\n"
            f"Hand off notes in EXACTLY this format:\n"
            f"## Background\n(bullets)\n## Ansatz/encoder comparison\n(markdown "
            f"table: name | structure | entanglement | pros | cons)\n"
            f"## Promising directions\n(ranked; each: the idea, why it should help "
            f"THIS task, what to try first)\nWhen you have enough, stop calling "
            f"tools and write the notes." + beat_note
        ),
        "generate": (
            f"You are an expert quantum circuit designer. Design the ENTANGLEMENT "
            f"PATTERN of a trainable quantum classification head (encoder + VQC) "
            f"for a {dim}-feature binary classification task, to MAXIMIZE test "
            f"accuracy at a LOW trainable-weight count -- reproducing the central "
            f"experiment of Cui & Huang (2024): stronger entanglement should "
            f"classify better, cheaply.\n"
            f"Strategy: (1) BUILD ON THE BEST -- start each session from the "
            f"current best head (first session: the strongest seed) and change ONE "
            f"thing about its ENTANGLEMENT: the encoder (product ZFeatureMap vs "
            f"entangling ZZFeatureMap), the ansatz's entanglement map "
            f"(linear/circular/full), its depth (reps), or the explicit CX "
            f"pattern. Compare the E1 (ZFeatureMap) and E2 (ZZFeatureMap) seeds to "
            f"see the effect, then push it. (2) WATCH OVERFITTING / BARREN "
            f"PLATEAUS -- the data is small and very deep/wide circuits train "
            f"poorly or overfit; if a bigger circuit scores WORSE, simplify.\n"
            + contract + beat_note + baseline_block + resource_block
        ),
        "review": (
            "You are a critical reviewer of trainable quantum classification "
            "heads. Given a candidate and its measured accuracy / f1 / weight "
            "count, give ONE specific, actionable change to raise accuracy without "
            "bloating trainable weights -- e.g. switch the encoder entanglement "
            "(Z vs ZZ), change the ansatz entanglement map, add/remove one "
            "reps layer, or restructure the CX pattern. If it errored, state the "
            "concrete fix." + beat_note + resource_note
        ),
        "deploy": (
            "You are writing the final report for an automated search over "
            "quantum classification heads. Given the best head and its stats, "
            "explain in a short paragraph what entanglement design was chosen, its "
            "final accuracy / f1 / weight count, and the rationale."
        ),
    }


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------
def make_stronglyentangled_task(dim: int = SE_DIM,
                                seed_stats: dict | None = None,
                                minimize_resources: bool = SE_MINIMIZE_RESOURCES,
                                ) -> DiscoveryTask:
    """Build the trainable-quantum-head DiscoveryTask.

    seed_stats: measured per-seed baselines shown to the agent as the numbers to
    beat. Default: SEED_BASELINE_STATS (empty until you run `baselines`); pass
    compute_seed_baselines(...) output for fresh values, or {} to disable.
    """
    if seed_stats is None:
        seed_stats = SEED_BASELINE_STATS
    # same evaluator (+ its data arg) the baselines use, so the two agree; the
    # full evaluator loads images itself, the frozen one takes cached features
    _eval_fn = active_evaluator()
    data = _baseline_data(dim)

    def _evaluate(code: str, *args, **kwargs) -> dict:
        kwargs.setdefault("plot_dir", os.path.join(task.run_dir, "plots"))
        kwargs.setdefault("n", dim)
        if minimize_resources:
            kwargs.setdefault("max_qubits", SE_MAX_QUBITS)
            kwargs.setdefault("max_gates", SE_MAX_GATES)
            kwargs.setdefault("max_depth", SE_MAX_DEPTH)
            kwargs.setdefault("max_weights", SE_MAX_WEIGHTS)
        return _eval_fn(code, data=data, *args, **kwargs)

    task = DiscoveryTask(
        name=f"stronglyentangled_qcnn_dim{dim}",
        system_prompts=_system_prompts(dim, seed_stats, minimize_resources),
        seeds={"quantum_head": _annotated_seeds(seed_stats)},
        evaluate_fn=_evaluate,
        knowledge_sources=[
            "Cui & Huang, Strong Entanglement Improves QNN Classification (ICONIP 2024)",
            "Cong et al., Quantum Convolutional Neural Networks (Nat. Phys. 2019)",
            "Havlicek et al., Supervised learning with quantum-enhanced feature spaces (arXiv:1804.11326)",
        ],
        documentation_source="qiskit + qiskit-machine-learning (EstimatorQNN / TorchConnector API)",
    )
    if minimize_resources:
        task.log("RESOURCE LIMITS (enforced)",
                 f"max_qubits={SE_MAX_QUBITS}  max_gates={SE_MAX_GATES}  "
                 f"max_depth={SE_MAX_DEPTH}  max_weights={SE_MAX_WEIGHTS}   "
                 f"(None = no limit; an over-budget head is scored but ok=false)")
    else:
        task.log("RESOURCE LIMITS", "SE_MINIMIZE_RESOURCES off -- no caps enforced")
    return task


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "baselines":
        run_dir = os.path.join("seed_baselines",
                               datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        compute_seed_baselines(plot_dir=run_dir)
        print("\nClassical baselines (same features):")
        print(format_classical_report(classical_baselines()))
    elif len(sys.argv) > 1 and sys.argv[1] == "classical":
        print(format_classical_report(classical_baselines()))
    else:
        task = make_stronglyentangled_task()
        print(task.describe())
        seed = next(iter(task.seeds("quantum_head").values()))
        print("baseline eval:", task.evaluate(seed, plot=True))
