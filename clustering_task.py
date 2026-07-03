"""
Concrete DiscoveryTask for DQC1 entropy-based clustering (mirrors the role of
qml_task.py + eval_harness.py, but for the unsupervised pipeline of dqc1.py /
"Unsupervised Quantum Machine Learning using One Clean Qubit").

The candidate artifact is the quantum FEATURE MAP used inside the DQC1
normalized-trace kernel. Everything else is reused from dqc1.py unchanged:
kernel construction (K_ij = |tr(U(x_i)U(x_j)^dag)|/2^n via the DQC1 Hadamard
test), the similarity map M = (K+1)/2, spectral compression + FABLE block
encoding, the DQC1 purity estimator for Renyi-2 entropy, DeltaH assignment,
and the Jenssen-style reduction / K* selection.

dqc1.py's env knobs still apply (INIT_CLUSTERS, N_INIT, DATA_NOISE,
DQC1_COMP_RANK, DQC1_EXACT_*, RANDOM_STATE, ...). New knobs:

    CLUSTER_DATASETS  which paper datasets to score on, "+"-separated
                      (default "spirals"; e.g. "spirals+moons+circles")
    CLUSTER_N_POINTS  points per dataset (default 60 -- the DQC1 kernel is
                      O(N^2) statevector simulations, keep this small)
    CLUSTER_VARIANT   "pre"  = init + DeltaH assignment only (DQC1-pre)
                      "full" = + reduction + K* selection    (DQC1-full)
"""

from __future__ import annotations

import os
import sys
import datetime

# dqc1.py prints non-ASCII (Δ, …) in its timers; Windows consoles default to
# cp1252, which would turn every evaluation into a UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: save figures, never open a window
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import dqc1
from dqc1 import (DQC1EntropyCluster, toy_sets, eval_scores,
                  labels_from_clusters, zero_mean_to_unit_interval)
from eval_harness import compile_feature_map
from DiscoveryTask import DiscoveryTask

CLUSTER_DATASETS = os.environ.get("CLUSTER_DATASETS", "spirals")
CLUSTER_N_POINTS = int(os.environ.get("CLUSTER_N_POINTS", 60))
CLUSTER_VARIANT = os.environ.get("CLUSTER_VARIANT", "pre").strip().lower()

# The DQC1 kernel circuit holds 2n+1 qubits (probe + system + purification),
# so cap the feature-map width to keep each kernel entry cheap to simulate.
MAX_QUBITS = 6


# ---------------------------------------------------------------------------
# Datasets: the paper's synthetic sets, straight from dqc1.toy_sets
# ---------------------------------------------------------------------------
def load_cluster_datasets(names: str = CLUSTER_DATASETS,
                          n_points: int = CLUSTER_N_POINTS) -> dict:
    """{name: (X, y)} for the requested subset of dqc1's synthetic datasets."""
    all_sets = {nm: (X, y) for (X, y, nm) in
                toy_sets(n=n_points, noise=None, random_state=dqc1.RANDOM_STATE)}
    requested = [s.strip() for s in names.split("+") if s.strip()]
    picked = {nm: all_sets[nm] for nm in requested if nm in all_sets}
    if not picked:
        raise ValueError(f"no valid dataset in {names!r}; "
                         f"available: {list(all_sets)}")
    return picked


# k-means reference score per dataset (K = #ground-truth classes), computed
# once and cached -- included in the metrics so the reviewer has a baseline.
_KMEANS_CACHE: dict = {}


def _kmeans_ari(name: str, X: np.ndarray, y: np.ndarray) -> float:
    if name not in _KMEANS_CACHE:
        Xn = zero_mean_to_unit_interval(X)
        labels = KMeans(n_clusters=len(np.unique(y)), n_init=20,
                        random_state=dqc1.RANDOM_STATE).fit_predict(Xn)
        _KMEANS_CACHE[name] = eval_scores(Xn, y, labels)["ARI"]
    return _KMEANS_CACHE[name]


# ---------------------------------------------------------------------------
# Plotting (dqc1.py panel style): one scatter per dataset, colored by labels
# ---------------------------------------------------------------------------
def _save_cluster_plot(results: dict, iteration=None, plot_dir: str = "plots"):
    os.makedirs(plot_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_iter{iteration}" if iteration is not None else ""
    path = os.path.join(plot_dir, f"clustering_{stamp}{tag}.png")
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5),
                             squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        X, labels = r["X"], r["labels"]
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=12, cmap="tab10")
        ax.set_title(f"{name} | K={r['scores']['K']} "
                     f"ARI={r['scores']['ARI']:.3f}", fontsize=10)
        ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# The evaluator: compile agent code -> run dqc1's clustering pipeline -> ARI
# ---------------------------------------------------------------------------
def evaluate_clustering(code: str, datasets: dict | None = None,
                        variant: str = CLUSTER_VARIANT, plot: bool = False,
                        iteration=None, plot_dir: str = "plots") -> dict:
    """Compile a feature map, cluster with the DQC1 pipeline, score with ARI.

    The candidate map is routed through dqc1.build_feature_map, so every
    kernel entry and purity estimate in dqc1.py uses the agent's circuit.
    Workers stay serial (kernel_workers=purity_workers=1): joblib subprocesses
    would re-import dqc1 and lose the patched feature map.
    """
    try:
        build_circuit = compile_feature_map(code)
        if datasets is None:
            datasets = load_cluster_datasets()

        probe = build_circuit(np.zeros(2))                # dry run / validation
        if probe.num_clbits:
            raise ValueError("circuit must not contain measurements")
        if probe.num_qubits > MAX_QUBITS:
            raise ValueError(f"use at most {MAX_QUBITS} qubits "
                             f"(got {probe.num_qubits}); the DQC1 kernel "
                             f"simulates 2n+1 qubits per entry")

        # route dqc1's kernel through the candidate map (kind/layers ignored)
        dqc1.build_feature_map = lambda x, kind, layers: build_circuit(
            np.asarray(x, dtype=float))

        results = {}
        for name, (X, y) in datasets.items():
            dqc = DQC1EntropyCluster(kernel_workers=1, purity_workers=1,
                                     gpu_ids=None)
            if variant == "full":
                dqc.fit_full(X)
                labels = dqc.labels_full_
            else:                                          # DQC1-pre
                dqc.fit(X)
                labels = labels_from_clusters(X.shape[0], dqc.clusters_)
            scores = eval_scores(zero_mean_to_unit_interval(X), y, labels)
            results[name] = {"X": X, "labels": labels, "scores": scores}

        metrics = {
            "ok": True,
            "ari": float(np.mean([r["scores"]["ARI"] for r in results.values()])),
            "nmi": float(np.mean([r["scores"]["NMI"] for r in results.values()])),
            "per_dataset": {
                name: {"ARI": round(r["scores"]["ARI"], 4),
                       "NMI": round(r["scores"]["NMI"], 4),
                       "K": int(r["scores"]["K"]),
                       "kmeans_ARI": round(_kmeans_ari(name, *datasets[name]), 4)}
                for name, r in results.items()},
            "variant": variant,
            "n_qubits": probe.num_qubits,
            "n_gates": int(sum(probe.count_ops().values())),
            "depth": probe.depth(),
        }
        if variant == "full":
            metrics["Kstar"] = {name: int(len(np.unique(
                r["labels"][r["labels"] >= 0]))) for name, r in results.items()}
        if plot:
            metrics["plot_path"] = _save_cluster_plot(results, iteration, plot_dir)
        return metrics
    except Exception as exc:  # returned to the agent for self-correction
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Seeds: the paper's five feature maps (dqc1.py), as build_circuit(x) strings
# under the same contract as agent code (L=2 layers baked in where relevant).
# ---------------------------------------------------------------------------
SEED_CLUSTER_FEATURE_MAPS = {
    # Single-Layer Pauli-Z Polynomial (karimi_feature_map)
    "karimi_single_layer_pauli_z": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n):
        qc.rz(2 * np.pi * x[i], i)
    for i in range(n):
        for j in range(i + 1, n):
            qc.rzz(2 * np.pi * x[i] * x[j], i, j)
    return qc
""",
    # Multi-Layer Pauli-Z Polynomial (fm_zz, layers=2)
    "zz_multi_layer": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(2):
        for q in range(n):
            qc.rz(2 * np.pi * x[q], q)
        for i in range(n):
            for j in range(i + 1, n):
                qc.rzz(2 * np.pi * x[i] * x[j], i, j)
    return qc
""",
    # Z-Diagonal Ising with CZ entanglers (fm_zdiag, layers=2)
    "zdiag_ising": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.rz(2 * np.pi * x[q], q)
        for q in range(0, n - 1, 2):
            qc.cz(q, q + 1)
        for q in range(1, n - 1, 2):
            qc.cz(q, q + 1)
    return qc
""",
    # Hardware-Efficient Data-Reuploading (fm_rxryrz, layers=2)
    "rxryrz_reuploading": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    for L in range(2):
        for q in range(n):
            a = x[q]
            qc.rx(2 * np.pi * (a + 0.1 * L), q)
            qc.ry(2 * np.pi * (a * a + 0.2 * L), q)
            qc.rz(2 * np.pi * (0.3 * a + 0.05 * L), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc
""",
    # Higher-Order Pauli-Z Polynomial (fm_pauli, layers=2)
    "pauli_higher_order": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.rz(2 * np.pi * (0.6 * x[q] + 0.2 * (x[q] ** 3)), q)
        for i in range(n):
            for j in range(i + 1, n):
                qc.rzz(2 * np.pi * (0.4 * x[i] * x[j]
                                    + 0.1 * (x[i] ** 2 + x[j] ** 2)), i, j)
    return qc
""",
}


# ---------------------------------------------------------------------------
# Prompts, one per agent role
# ---------------------------------------------------------------------------
def _system_prompts(dataset_names: list[str], variant: str) -> dict[str, str]:
    sets = ", ".join(dataset_names)
    return {
        "explore": (
            f"You are a research assistant for designing quantum feature maps for "
            f"UNSUPERVISED entropy-based clustering with the DQC1 (one clean qubit) "
            f"model. The kernel is the DQC1 normalized trace "
            f"K(x,x') = |tr(U(x)U(x')^dag)|/2^n, mapped to a similarity M=(K+1)/2, "
            f"and clusters are grown by minimizing the increase in Renyi-2 entropy "
            f"H2 = -log Tr(rho^2) estimated via DQC1 purity. Target datasets: "
            f"{sets} (2D, non-convex geometry). Gather encoding strategies that make "
            f"the trace kernel separate curved manifolds (polynomial Pauli-Z phases, "
            f"data re-uploading, radial/angular encodings). Take concise, actionable "
            f"notes for the designer, then stop calling tools and summarize."
        ),
        "generate": (
            f"You are an expert quantum circuit designer. Design the quantum feature "
            f"map U(x) used inside a DQC1 normalized-trace kernel "
            f"K(x,x') = |tr(U(x)U(x')^dag)|/2^n for UNSUPERVISED entropy-based "
            f"clustering (Renyi-2, DQC1-{variant}) on these 2D synthetic datasets: "
            f"{sets}. MAXIMIZE the mean Adjusted Rand Index (ARI) against the "
            f"ground-truth classes.\n"
            f"Key physics: only the relative unitary U(x)U(x')^dag matters, so "
            f"global structure comes from how phases depend on x. Diagonal-only "
            f"maps give K = product of cosines of phase differences; non-diagonal "
            f"rotations (RX/RY) change this qualitatively.\n"
            f"Contract for the code you return:\n"
            f"- Define exactly one function build_circuit(x) -> QuantumCircuit.\n"
            f"- x is a length-2 numpy array with entries normalized to about "
            f"[-1, 1]. Use at least 2 qubits, at most {MAX_QUBITS} (the DQC1 "
            f"kernel simulates 2n+1 qubits per entry, so stay small).\n"
            f"- NO imports; `np` and `QuantumCircuit` are in scope. No "
            f"measurements, no trainable parameters.\n"
            f"- Present each candidate as a ```python code block, THEN call the "
            f"`evaluate` tool with that exact code to score it (it runs the full "
            f"DQC1 clustering pipeline; each call takes a while, so choose "
            f"variants deliberately). Metrics include per-dataset ARI/NMI and a "
            f"k-means reference ARI. After you see the scores, refine and "
            f"evaluate again, or reply WITHOUT a tool call when you are satisfied."
        ),
        "review": (
            "You are a critical reviewer of quantum feature maps for DQC1 "
            "entropy-based clustering. Given a candidate and its measured ARI/NMI "
            "per dataset (with a k-means reference), give ONE specific, actionable "
            "improvement -- e.g. phase scaling (kernel bandwidth of the trace "
            "kernel), higher-order polynomial Z terms, non-diagonal rotations, "
            "radial/angular encodings for rings, or fewer layers if the kernel "
            "is concentrating. Two sentences max. If it errored, state the fix."
        ),
        "deploy": (
            "You are writing the final report for an automated search over quantum "
            "feature maps for DQC1 entropy-based clustering. Given the best map and "
            "its stats, explain in a short paragraph what was designed, its final "
            "per-dataset ARI/NMI, and the design rationale."
        ),
    }


def make_clustering_task(datasets: str = CLUSTER_DATASETS,
                         n_points: int = CLUSTER_N_POINTS,
                         variant: str = CLUSTER_VARIANT) -> DiscoveryTask:
    data = load_cluster_datasets(datasets, n_points)

    def _evaluate(code: str, *args, **kwargs) -> dict:
        return evaluate_clustering(code, datasets=data, variant=variant,
                                   *args, **kwargs)

    return DiscoveryTask(
        name=f"dqc1_clustering_{'_'.join(data)}_{variant}",
        system_prompts=_system_prompts(list(data), variant),
        seeds={"cluster_feature_map": SEED_CLUSTER_FEATURE_MAPS},
        evaluate_fn=_evaluate,
        knowledge_sources=[
            "Salehi et al., Unsupervised QML using One Clean Qubit (SPIE)",
            "Jenssen et al., Clustering using Renyi's entropy (IJCNN 2003)",
            "arXiv:2210.09275 (Karimi, power of one clean qubit in supervised ML)",
        ],
        documentation_source="qiskit (QuantumCircuit API)",
    )


if __name__ == "__main__":
    task = make_clustering_task()
    print(task.describe())
    seed = SEED_CLUSTER_FEATURE_MAPS["karimi_single_layer_pauli_z"]
    print("baseline eval:", task.evaluate(seed, plot=True))
