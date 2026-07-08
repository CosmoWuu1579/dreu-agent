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
    CLUSTER_N_POINTS  points per dataset (default 100 -- the DQC1 kernel is
                      O(N^2) statevector simulations, keep this modest)
    CLUSTER_VARIANT   "full" (default) = init + DeltaH + reduction + K*
                      "pre"            = init + DeltaH assignment only

    CAUTION on "pre": at small N with the default INIT_CLUSTERS/N_INIT,
    dqc1.py's seed-growth loop (pure Euclidean nearest-neighbor) can consume
    every point before the kernel-dependent DeltaH step runs -- then the score
    is IDENTICAL for every feature map and the search objective is flat
    (verified at N=24 and N=60). "full" always depends on the kernel through
    the reduction/reassignment step, so it is the default for discovery.
"""

from __future__ import annotations

import os
import sys
import datetime

# Honor .env even when THIS module is the entry point (e.g.
# `python clustering_task.py baselines`) -- otherwise only run_discovery_
# clustering.py loads it and a direct run silently uses defaults. Must run
# before `import dqc1` and the CLUSTER_*/DQC1_* reads below, since those read
# os.environ at import. load_dotenv does NOT override already-set vars, so it
# is a harmless no-op when run_discovery already called it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
CLUSTER_N_POINTS = int(os.environ.get("CLUSTER_N_POINTS", 100))
CLUSTER_VARIANT = os.environ.get("CLUSTER_VARIANT", "full").strip().lower()
# 1 -> the generate/review prompts also push for fewer qubits/gates/lower
# depth as a SECONDARY objective (ARI always stays primary)
CLUSTER_MINIMIZE_RESOURCES = os.environ.get("CLUSTER_MINIMIZE_RESOURCES", "0") != "0"


def _limit(name: str) -> int | None:
    """A hard resource cap from the env: unset / "" / "0" -> None (no limit)."""
    v = os.environ.get(name, "").strip()
    return int(v) if v and v != "0" else None


# HARD resource budgets, ONLY enforced when CLUSTER_MINIMIZE_RESOURCES is on. A
# circuit that busts any of these is marked ok=False ("not viable") so it cannot
# win. Unlike the supervised task (fast eval), clustering is minutes per run, so
# an over-budget circuit is REJECTED UP FRONT (before the O(N^2) pipeline) --
# the agent still gets the qubit/gate/depth counts and the "shrink it" message.
# Set them in .env, e.g. CLUSTER_MAX_QUBITS=4.
CLUSTER_MAX_QUBITS = _limit("CLUSTER_MAX_QUBITS")
CLUSTER_MAX_GATES = _limit("CLUSTER_MAX_GATES")
CLUSTER_MAX_DEPTH = _limit("CLUSTER_MAX_DEPTH")

# The DQC1 kernel circuit holds 2n+1 qubits (probe + system + purification), so
# cap the feature-map width to keep each kernel entry cheap to simulate. This is
# a HARD computational ceiling, always enforced (raises), separate from the
# optional CLUSTER_MAX_QUBITS economy cap above.
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
# Config snapshot: a config.txt written into every folder that receives plots,
# recording every knob that affects a clustering result -- so any figure can be
# reproduced from the file sitting next to it. dqc1 values are read LIVE (they
# may be overridden at runtime), not from import-time copies.
# ---------------------------------------------------------------------------
def pipeline_config(**extra) -> dict:
    """Every setting that changes a clustering result, as an ordered dict."""
    cfg = {
        # this module's knobs
        "CLUSTER_DATASETS": CLUSTER_DATASETS,
        "CLUSTER_N_POINTS": CLUSTER_N_POINTS,
        "CLUSTER_VARIANT": CLUSTER_VARIANT,
        "CLUSTER_MINIMIZE_RESOURCES": CLUSTER_MINIMIZE_RESOURCES,
        # hard resource caps (None = unlimited; only enforced when minimize on)
        "CLUSTER_MAX_QUBITS": CLUSTER_MAX_QUBITS,
        "CLUSTER_MAX_GATES": CLUSTER_MAX_GATES,
        "CLUSTER_MAX_DEPTH": CLUSTER_MAX_DEPTH,
        "MAX_QUBITS": MAX_QUBITS,
        # dqc1.py knobs that drive the kernel / entropy / clustering
        "DATA_NOISE": dqc1.DATA_NOISE,
        "RANDOM_STATE": dqc1.RANDOM_STATE,
        "INIT_CLUSTERS": dqc1.INIT_CLUSTERS,
        "MIN_CLUSTERS": dqc1.MIN_CLUSTERS,
        "N_INIT": dqc1.N_INIT,
        "REPEATS": dqc1.REPEATS,
        "DQC1_COMP_RANK": dqc1.DQC1_COMP_RANK,
        "DQC1_EXACT_KERNEL": dqc1.DQC1_EXACT_KERNEL,
        "DQC1_EXACT_PURITY": dqc1.DQC1_EXACT_PURITY,
        "DQC1_KERNEL_SHOTS": dqc1.DQC1_KERNEL_SHOTS,
        "DQC1_PURITY_SHOTS": dqc1.DQC1_PURITY_SHOTS,
        "PARZEN_SIGMA_GRID": dqc1.PARZEN_SIGMA_GRID,
    }
    cfg.update(extra)                      # caller extras (e.g. which map)
    return cfg


def write_plot_config(plot_dir: str, **extra) -> str:
    """Write/refresh config.txt in plot_dir, snapshotting the settings behind
    the figures there. Note: the candidate/seed feature map is supplied as
    build_circuit code, so dqc1's FEATURE_MAP/LAYERS do NOT apply here."""
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "config.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# clustering config -- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("# feature map is candidate/seed build_circuit code; "
                "dqc1 FEATURE_MAP/LAYERS are not used\n")
        for k, v in pipeline_config(**extra).items():
            f.write(f"{k} = {v}\n")
    return path


# ---------------------------------------------------------------------------
# Plotting (dqc1.py panel style): one scatter per dataset, colored by labels
# ---------------------------------------------------------------------------
def _save_cluster_plot(results: dict, iteration=None, plot_dir: str = "plots"):
    os.makedirs(plot_dir, exist_ok=True)
    write_plot_config(plot_dir)
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
                        iteration=None, plot_dir: str = "plots",
                        max_qubits: int | None = None,
                        max_gates: int | None = None,
                        max_depth: int | None = None) -> dict:
    """Compile a feature map, cluster with the DQC1 pipeline, score with ARI.

    The candidate map is routed through dqc1.build_feature_map, so every
    kernel entry and purity estimate in dqc1.py uses the agent's circuit.
    Workers stay serial (kernel_workers=purity_workers=1): joblib subprocesses
    would re-import dqc1 and lose the patched feature map.

    max_qubits / max_gates / max_depth: HARD resource budgets (None = no
        limit). A circuit that exceeds any set limit is REJECTED UP FRONT with
        ok=False + a "not viable" error (the O(N^2) clustering is NOT run --
        it's minutes of compute a losing circuit does not deserve), but the
        result still carries the qubit/gate/depth counts and the evaluate call
        still counts. Wired by clustering_task only when
        CLUSTER_MINIMIZE_RESOURCES is on.
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

        # HARD resource budget: reject before the expensive clustering, but
        # report the counts + violation so the agent knows exactly what to cut.
        n_qubits = probe.num_qubits
        n_gates = int(sum(probe.count_ops().values()))
        depth = probe.depth()
        violations = []
        if max_qubits is not None and n_qubits > max_qubits:
            violations.append(f"qubits {n_qubits} > limit {max_qubits}")
        if max_gates is not None and n_gates > max_gates:
            violations.append(f"gates {n_gates} > limit {max_gates}")
        if max_depth is not None and depth > max_depth:
            violations.append(f"depth {depth} > limit {max_depth}")
        if violations:
            return {
                "ok": False,
                "resource_violation": violations,
                "n_qubits": n_qubits, "n_gates": n_gates, "depth": depth,
                "variant": variant,
                "error": ("NOT VIABLE -- circuit exceeds the resource budget: "
                          + "; ".join(violations) + ". It was NOT clustered "
                          "(over-budget circuits are rejected before the "
                          "expensive pipeline) and cannot be submitted. Reduce "
                          "qubits/gates/depth and try a smaller circuit."),
            }

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
    # ---- new maps from new_feature_maps.py (Colab sweep), L=1 as used there --
    # Pauli-XZ (fm_pauli_xz): RX/RZ(x^2) fields + X_i Z_j coupling via H-RZZ-H
    "pauli_xz": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n):
        qc.rx(2 * np.pi * x[i], i)
        qc.rz(2 * np.pi * x[i] ** 2, i)
    for i in range(n):
        for j in range(i + 1, n):
            qc.h(i)
            qc.rzz(2 * np.pi * x[i] * x[j], i, j)
            qc.h(i)
    return qc
""",
    # IQP-style (fm_iqp): H layer inside the loop, then Z + ZZ phases
    "iqp": """
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
    # Trotterized Hamiltonian (fm_hamiltonian):
    # H(x) = sum_i x_i Z_i + x_i^2 X_i + x_i x_{i+1} Z_i Z_{i+1}
    "hamiltonian_trotter": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.rz(2 * np.pi * x[i], i)
        qc.rx(2 * np.pi * x[i] ** 2, i)
    for i in range(n - 1):
        qc.rzz(2 * np.pi * x[i] * x[i + 1], i, i + 1)
    return qc
""",
    # Random kitchen sinks (fm_random_kitchen_sinks): fixed-seed random
    # frequencies/phases per qubit -> deterministic across calls
    "random_kitchen_sinks": """
def build_circuit(x):
    n = len(x)
    rng = np.random.default_rng(123)
    w_rx = rng.normal(size=n); b_rx = rng.uniform(0, 2 * np.pi, size=n)
    w_ry = rng.normal(size=n); b_ry = rng.uniform(0, 2 * np.pi, size=n)
    w_rz = rng.normal(size=n); b_rz = rng.uniform(0, 2 * np.pi, size=n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.rx(2 * np.pi * w_rx[i] * x[i] + b_rx[i], i)
        qc.ry(2 * np.pi * w_ry[i] * x[i] + b_ry[i], i)
        qc.rz(2 * np.pi * w_rz[i] * x[i] + b_rz[i], i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc
""",
}


# ---------------------------------------------------------------------------
# Measured seed baselines: shown to the agent (in the seed library AND the
# generate system prompt) as "the numbers to beat".
#
# Two ways to refresh when CLUSTER_* / dqc1 knobs change:
#   manual : run  python clustering_task.py baselines  and paste the printed
#            dict over SEED_BASELINE_STATS below, or
#   dynamic: pass make_clustering_task(seed_stats=compute_seed_baselines(...))
# Values below measured at N=100 (seed_baselines/20260706_171309). NOTE: the
# DQC1-full pipeline is strongly N-sensitive with a FIXED DQC1_COMP_RANK -- at
# N=100 the rank-8 cluster compression blurs the entropy estimate, so scores
# collapse vs N=60 (rxryrz 0.634->0.060) and the ranking inverts (karimi/iqp
# lead at 0.186). If DQC1_COMP_RANK or N changes, refresh these.
# ---------------------------------------------------------------------------
SEED_BASELINE_CONFIG = "spirals, DQC1-full, N=80, DATA_NOISE=0.0, RANDOM_STATE=0"
SEED_BASELINE_STATS = {
    "karimi_single_layer_pauli_z": {"ari": 0.2168, "nmi": 0.204, "qubits": 2, "gates": 5, "depth": 3},
    "zz_multi_layer": {"ari": 0.0394, "nmi": 0.0411, "qubits": 2, "gates": 8, "depth": 5},
    "zdiag_ising": {"ari": -0.0036, "nmi": 0.0063, "qubits": 2, "gates": 6, "depth": 4},
    "rxryrz_reuploading": {"ari": 0.0398, "nmi": 0.0431, "qubits": 2, "gates": 14, "depth": 8},
    "pauli_higher_order": {"ari": -0.0011, "nmi": 0.0086, "qubits": 2, "gates": 6, "depth": 4},
    "pauli_xz": {"ari": 0.1519, "nmi": 0.1896, "qubits": 2, "gates": 9, "depth": 6},
    "iqp": {"ari": 0.2168, "nmi": 0.204, "qubits": 2, "gates": 5, "depth": 3},
    "hamiltonian_trotter": {"ari": 0.1519, "nmi": 0.1896, "qubits": 2, "gates": 5, "depth": 3},
    "random_kitchen_sinks": {"ari": -0.0098, "nmi": 0.0006, "qubits": 2, "gates": 7, "depth": 4},
}


def _annotated_seeds(seed_stats: dict | None) -> dict:
    """The seed library with each map's measured baseline prepended as a
    comment, so `view_seed_library` shows code AND score together."""
    if not seed_stats:
        return dict(SEED_CLUSTER_FEATURE_MAPS)
    out = {}
    for name, code in SEED_CLUSTER_FEATURE_MAPS.items():
        s = seed_stats.get(name)
        if isinstance(s, dict) and "ari" in s:
            header = (f"# measured ({SEED_BASELINE_CONFIG}): "
                      f"mean ARI={s['ari']:.3f}, NMI={s['nmi']:.3f}, "
                      f"qubits={s.get('qubits', '?')}, gates={s['gates']}, "
                      f"depth={s['depth']}")
        else:
            header = "# (no measured baseline yet)"
        out[name] = f"\n{header}\n{code.strip()}\n"
    return out


def save_seed_comparison_chart(stats: dict, path: str, note: str = "") -> str:
    """Horizontal grouped bars of mean ARI/NMI per feature map, sorted by ARI
    (best on top); rows whose name starts with '>>' (the agent) are bolded.
    `stats` maps name -> {"ari", "nmi", ...}; error entries are skipped."""
    rows = [(n, s) for n, s in stats.items()
            if isinstance(s, dict) and "ari" in s]
    if not rows:
        return ""
    write_plot_config(os.path.dirname(path) or ".", chart_note=note)
    rows.sort(key=lambda kv: kv[1]["ari"])            # barh: last row on top
    names = [n for n, _ in rows]
    y = np.arange(len(rows))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 2))
    b_ari = ax.barh(y + h / 2, [s["ari"] for _, s in rows], height=h,
                    color="#2563eb", label="mean ARI")
    b_nmi = ax.barh(y - h / 2, [s["nmi"] for _, s in rows], height=h,
                    color="#d97706", label="mean NMI")
    ax.bar_label(b_ari, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b_nmi, fmt="%.3f", padding=3, fontsize=8)
    ax.set_yticks(y, names, fontsize=9)
    for tick, name in zip(ax.get_yticklabels(), names):
        if name.startswith(">>"):
            tick.set_fontweight("bold")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("score")
    ax.set_title("Feature-map comparison" + (f" — {note}" if note else ""),
                 fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def new_baseline_run_dir(root: str = "seed_baselines") -> str:
    """Create and return seed_baselines/<timestamp>/ -- one folder per script
    run, so each invocation's figures stay together instead of piling up (and
    overwriting each other's seed_comparison.png) in the shared root."""
    path = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    return path


def compute_seed_baselines(datasets: str = CLUSTER_DATASETS,
                           n_points: int = CLUSTER_N_POINTS,
                           variant: str = CLUSTER_VARIANT,
                           plot_dir: str | None = "seed_baselines") -> dict:
    """Score every seed feature map with the real evaluator (~1 min each) and
    return {name: {"ari", "nmi", "gates", "depth"}}.

    Also prints the dict as paste-ready Python for SEED_BASELINE_STATS, and
    (with plot_dir) saves one clustering figure per seed plus the comparison
    bar chart. Use whenever the datasets / pipeline knobs change.
    """
    data = load_cluster_datasets(datasets, n_points)
    config = (f"{datasets}, DQC1-{variant}, N={n_points}, "
              f"DATA_NOISE={dqc1.DATA_NOISE}, RANDOM_STATE={dqc1.RANDOM_STATE}")
    stats: dict = {}
    for name, code in SEED_CLUSTER_FEATURE_MAPS.items():
        m = evaluate_clustering(code, datasets=data, variant=variant,
                                plot=bool(plot_dir), iteration=f"seed_{name}",
                                plot_dir=plot_dir or "plots")
        if m.get("ok"):
            stats[name] = {"ari": round(m["ari"], 4), "nmi": round(m["nmi"], 4),
                           "qubits": m["n_qubits"], "gates": m["n_gates"],
                           "depth": m["depth"]}
            print(f"  {name:34s} ari={m['ari']:.4f} nmi={m['nmi']:.4f} "
                  f"qubits={m['n_qubits']} gates={m['n_gates']} depth={m['depth']}")
        else:
            stats[name] = {"error": m.get("error")}
            print(f"  {name:34s} ERROR: {m.get('error')}")
    if plot_dir:
        chart = save_seed_comparison_chart(
            stats, os.path.join(plot_dir, "seed_comparison.png"), note=config)
        print(f"  [chart] {chart}")
    print("\n# paste over SEED_BASELINE_STATS in clustering_task.py:")
    print(f'SEED_BASELINE_CONFIG = "{config}"')
    print("SEED_BASELINE_STATS = {")
    for n, s in stats.items():
        if "ari" in s:
            print(f'    "{n}": {{"ari": {s["ari"]}, "nmi": {s["nmi"]}, '
                  f'"qubits": {s["qubits"]}, "gates": {s["gates"]}, '
                  f'"depth": {s["depth"]}}},')
    print("}")
    return stats


def classical_baselines(datasets: dict | None = None,
                        plot_dir: str | None = None) -> dict:
    """Score dqc1.py's classical methods on the same datasets: Parzen FULL
    (Gaussian Renyi-2 via dqc1.parzen_full_labels, sigma grid + reduction +
    K*) and k-means (K = #ground-truth classes). With plot_dir, saves a
    dqc1.run_synthetic_and_save-style panel figure (one row per dataset,
    Parzen | K-Means). Returns {dataset: {"parzen_full": ..., "kmeans": ...}}.
    """
    if datasets is None:
        datasets = load_cluster_datasets()
    out: dict = {}
    panels: dict = {}
    for name, (X, y) in datasets.items():
        Xn = zero_mean_to_unit_interval(X)
        p_labels, p_kstar, p_sigma = dqc1.parzen_full_labels(
            X, sigma_grid=dqc1.PARZEN_SIGMA_GRID, repeats=dqc1.REPEATS)
        p = eval_scores(Xn, y, p_labels)
        km_labels = KMeans(n_clusters=len(np.unique(y)), n_init=20,
                           random_state=dqc1.RANDOM_STATE).fit_predict(Xn)
        k = eval_scores(Xn, y, km_labels)
        out[name] = {
            "parzen_full": {"ARI": round(p["ARI"], 4), "NMI": round(p["NMI"], 4),
                            "Kstar": int(p_kstar), "sigma": float(p_sigma)},
            "kmeans": {"ARI": round(k["ARI"], 4), "NMI": round(k["NMI"], 4),
                       "K": int(k["K"])},
        }
        panels[name] = (X, p_labels, km_labels)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        write_plot_config(plot_dir, methods="Parzen FULL (Gaussian) + k-means")
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(plot_dir, f"classical_baselines_{stamp}.png")
        fig, axes = plt.subplots(len(panels), 2,
                                 figsize=(10, 4.5 * len(panels)), squeeze=False)
        for row, (name, (X, p_labels, km_labels)) in enumerate(panels.items()):
            r = out[name]
            ax1, ax2 = axes[row]
            ax1.scatter(X[:, 0], X[:, 1], c=p_labels, s=12, cmap="tab10")
            ax1.set_title(f"Parzen FULL (Gauss) — {name} "
                          f"(K*={r['parzen_full']['Kstar']}, "
                          f"σ={r['parzen_full']['sigma']:.2f}, "
                          f"ARI={r['parzen_full']['ARI']:.3f})", fontsize=10)
            ax2.scatter(X[:, 0], X[:, 1], c=km_labels, s=12, cmap="tab10")
            ax2.set_title(f"K-Means — {name} (K={r['kmeans']['K']}, "
                          f"ARI={r['kmeans']['ARI']:.3f})", fontsize=10)
            for ax in (ax1, ax2):
                ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out["plot_path"] = path
    return out


def format_classical_report(res: dict) -> str:
    """classical_baselines() output as a printable table (one row/dataset)."""
    lines = []
    for name, r in res.items():
        if name == "plot_path":
            continue
        p, k = r["parzen_full"], r["kmeans"]
        lines.append(
            f"  {name:16s} Parzen-FULL: ARI={p['ARI']:.3f} NMI={p['NMI']:.3f} "
            f"(K*={p['Kstar']}, sigma={p['sigma']:.2f}) | "
            f"k-means: ARI={k['ARI']:.3f} NMI={k['NMI']:.3f} (K={k['K']})")
    if res.get("plot_path"):
        lines.append(f"  [plot] {res['plot_path']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts, one per agent role
# ---------------------------------------------------------------------------
def _system_prompts(dataset_names: list[str], variant: str,
                    seed_stats: dict | None = None,
                    minimize_resources: bool = False) -> dict[str, str]:
    sets = ", ".join(dataset_names)
    # effective upper qubit bound stated in the contract: the hard MAX_QUBITS
    # ceiling, tightened to CLUSTER_MAX_QUBITS when that economy cap is set
    q_hi = MAX_QUBITS
    if minimize_resources and CLUSTER_MAX_QUBITS is not None:
        q_hi = min(MAX_QUBITS, CLUSTER_MAX_QUBITS)
    # optional secondary objective: circuit economy (toggle, off by default)
    resource_block, resource_note = "", ""
    if minimize_resources:
        # hard, enforced budgets (only the ones actually set in .env)
        limits = []
        if CLUSTER_MAX_QUBITS is not None:
            limits.append(f"at most {CLUSTER_MAX_QUBITS} qubits")
        if CLUSTER_MAX_GATES is not None:
            limits.append(f"at most {CLUSTER_MAX_GATES} gates")
        if CLUSTER_MAX_DEPTH is not None:
            limits.append(f"depth at most {CLUSTER_MAX_DEPTH}")
        limit_line = ""
        if limits:
            limit_line = (
                "\nHARD RESOURCE LIMITS (ENFORCED): your circuit must use "
                + ", ".join(limits) + ". A circuit that exceeds ANY of these is "
                "REJECTED before clustering and marked NOT VIABLE (ok=false) no "
                "matter what -- it cannot be submitted or win, and the evaluate "
                "result will say by how much it went over. Stay within budget.")
        resource_block = (
            "\n\n# Secondary objective: circuit economy\n"
            "Among candidates with comparable mean ARI, PREFER fewer qubits, "
            "fewer gates, and lower depth (all three are reported in every "
            "evaluation result). Never trade ARI away for economy: beat the "
            "baseline first, then simplify -- drop gates that do not change "
            "the score, use the fewest qubits that work, and prefer shallow "
            "layers over stacked ones."
            + limit_line)
        resource_note = (" Also flag waste: qubits, gates, or depth that do "
                         "not earn their ARI.")
    # "numbers to beat", built from the measured seed baselines (if provided)
    rows = [(n, s) for n, s in (seed_stats or {}).items()
            if isinstance(s, dict) and "ari" in s]
    beat_note, baseline_block = "", ""
    if rows:
        best_name, best_s = max(rows, key=lambda kv: kv[1]["ari"])
        table = "\n".join(f"  - {n}: mean ARI={s['ari']:.3f}, "
                          f"NMI={s['nmi']:.3f}, qubits={s.get('qubits', '?')}, "
                          f"gates={s['gates']}, depth={s['depth']}"
                          for n, s in rows)
        beat_note = (f" The measured best seed is {best_name} at mean ARI "
                     f"{best_s['ari']:.3f}; the goal is to EXCEED it.")
        baseline_block = (
            f"\n\n# Seed baselines to beat ({SEED_BASELINE_CONFIG})\n{table}\n"
            f"NUMBER TO BEAT: {best_name} at mean ARI {best_s['ari']:.3f} -- "
            f"success = mean ARI strictly ABOVE it. First reason about why the "
            f"strong seeds win and weak ones fail, then extend/hybridize/rethink "
            f"them.\nSUBMISSION RULE: submit your OWN circuit, never a seed "
            f"verbatim (nor one with only cosmetic edits). Even if nothing beat "
            f"{best_name}, submit your best original and say it fell short -- "
            f"returning a seed is an automatic failure.")
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
            + beat_note
        ),
        "generate": (
            f"You are an expert quantum circuit designer. Design the feature map "
            f"U(x) inside a DQC1 normalized-trace kernel "
            f"K(x,x')=|tr(U(x)U(x')^dag)|/2^n for unsupervised Renyi-2 entropy "
            f"clustering (DQC1-{variant}) on 2D datasets: {sets}. MAXIMIZE mean "
            f"ARI vs ground truth.\n"
            f"Physics: only the relative unitary U(x)U(x')^dag matters. "
            f"Diagonal-only maps give K = product of cosines of phase diffs; "
            f"RX/RY rotations change this qualitatively.\n"
            f"Strategy: (1) LOCAL SEARCH FIRST -- your first trial each session is "
            f"the best seed with ONE change (bandwidth scalar, a layer, a swapped "
            f"feature); only redesign after scoring that. (2) SMALL BEATS BIG -- "
            f"the trace kernel CONCENTRATES toward zero as circuits grow, "
            f"flattening K and killing separability; prefer 2-3 qubits, <=2 "
            f"layers. A bigger circuit scoring WORSE means concentration -- "
            f"shrink, don't grow.\n"
            f"Contract: define exactly one build_circuit(x)->QuantumCircuit; x is "
            f"a length-2 array ~[-1,1]; 2-{q_hi} qubits (the kernel "
            f"simulates 2n+1 per entry); no imports (`np`, `QuantumCircuit` in "
            f"scope), no measurements or trainable params. Present each candidate "
            f"as a ```python block, THEN call `evaluate` on it (runs the full "
            f"pipeline, slow -- choose deliberately; metrics include per-dataset "
            f"ARI/NMI + a k-means reference). Refine from the scores, or reply "
            f"WITHOUT a tool call to submit."
            + baseline_block + resource_block
        ),
        "review": (
            "You are a critical reviewer of quantum feature maps for DQC1 "
            "entropy-based clustering. Given a candidate and its measured ARI/NMI "
            "per dataset (with a k-means reference), give ONE specific, actionable "
            "improvement -- e.g. phase scaling (kernel bandwidth of the trace "
            "kernel), higher-order polynomial Z terms, non-diagonal rotations, "
            "radial/angular encodings for rings, or fewer layers if the kernel "
            "is concentrating. Be concise but give as much detail as the "
            "improvement warrants. If it errored, state the fix."
            + beat_note + resource_note
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
                         variant: str = CLUSTER_VARIANT,
                         seed_stats: dict | None = None,
                         minimize_resources: bool = CLUSTER_MINIMIZE_RESOURCES,
                         ) -> DiscoveryTask:
    """seed_stats: measured per-seed baselines shown to the agent as the
    numbers to beat. Default: the hardcoded SEED_BASELINE_STATS (measured
    under SEED_BASELINE_CONFIG -- refresh if the config changed); pass
    compute_seed_baselines(...) output for fresh values, or {} to disable.
    minimize_resources: adds the optional circuit-economy section to the
    generate/review prompts (env toggle: CLUSTER_MINIMIZE_RESOURCES=1)."""
    if seed_stats is None:
        seed_stats = SEED_BASELINE_STATS
    data = load_cluster_datasets(datasets, n_points)

    def _evaluate(code: str, *args, **kwargs) -> dict:
        # figures live in the task's per-run folder, next to the transcript
        # (`task` is bound below, before the first evaluation runs); plots/
        # subfolder matches qml_task.py's convention
        kwargs.setdefault("plot_dir", os.path.join(task.run_dir, "plots"))
        # enforce the hard resource budget ONLY when minimize_resources is on
        if minimize_resources:
            kwargs.setdefault("max_qubits", CLUSTER_MAX_QUBITS)
            kwargs.setdefault("max_gates", CLUSTER_MAX_GATES)
            kwargs.setdefault("max_depth", CLUSTER_MAX_DEPTH)
        return evaluate_clustering(code, datasets=data, variant=variant,
                                   *args, **kwargs)

    task = DiscoveryTask(
        name=f"dqc1_clustering_{'_'.join(data)}_{variant}",
        system_prompts=_system_prompts(list(data), variant, seed_stats,
                                       minimize_resources),
        seeds={"cluster_feature_map": _annotated_seeds(seed_stats)},
        evaluate_fn=_evaluate,
        knowledge_sources=[
            "Salehi et al., Unsupervised QML using One Clean Qubit (SPIE)",
            "Jenssen et al., Clustering using Renyi's entropy (IJCNN 2003)",
            "arXiv:2210.09275 (Karimi, power of one clean qubit in supervised ML)",
        ],
        documentation_source="qiskit (QuantumCircuit API)",
    )
    # record the resource-limit state in the run transcript (also snapshotted
    # into each config.txt via pipeline_config)
    if minimize_resources:
        task.log("RESOURCE LIMITS (enforced)",
                 f"max_qubits={CLUSTER_MAX_QUBITS}  max_gates={CLUSTER_MAX_GATES}"
                 f"  max_depth={CLUSTER_MAX_DEPTH}   (None = no limit; an "
                 f"over-budget circuit is rejected before clustering, marked "
                 f"ok=false so it cannot win)")
    else:
        task.log("RESOURCE LIMITS",
                 "CLUSTER_MINIMIZE_RESOURCES off -- no caps enforced")
    return task


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "baselines":
        # refresh the seed baselines (~1 min per seed at the default 60
        # points): per-seed clustering figures + comparison chart + paste-
        # ready stats dict, THEN the classical methods (Parzen FULL Gaussian
        # + k-means) with their own clustering panels -- everything lands in
        # ONE timestamped seed_baselines/<stamp>/ folder so quantum and
        # classical plots for this run sit side by side
        run_dir = new_baseline_run_dir()
        compute_seed_baselines(plot_dir=run_dir)
        print("\nClassical baselines (same datasets):")
        print(format_classical_report(classical_baselines(plot_dir=run_dir)))
    elif len(sys.argv) > 1 and sys.argv[1] == "classical":
        # classical methods only (fast: no quantum circuits involved)
        print(format_classical_report(
            classical_baselines(plot_dir=new_baseline_run_dir())))
    else:
        task = make_clustering_task()
        print(task.describe())
        seed = SEED_CLUSTER_FEATURE_MAPS["karimi_single_layer_pauli_z"]
        print("baseline eval:", task.evaluate(seed, plot=True))
