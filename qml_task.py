"""
Concrete DiscoveryTask for supervised quantum feature-map design (mirrors the
role of eval_harness.py, wrapped as a reusable DiscoveryTask). It plugs the
fidelity / DQC1 quantum-kernel SVM evaluator, seed library, and prompts into the
generic framework -- the SUPERVISED sibling of clustering_task.py, sharing its
baseline / prompt / plotting machinery but with a different dataset and
evaluator (SVM test accuracy instead of clustering ARI).

Build one with make_qml_task(dim=..., kernel=...) and hand it to an
AgentPipeline. Env knobs (also honored when this file is run directly, since it
loads .env itself):

    AGENT_DIM               number of features / min qubits (default 2)
    AGENT_KERNEL            "fidelity" (scalable search objective) | "dqc1"
    QML_MINIMIZE_RESOURCES  1 -> also push for fewer qubits/gates/lower depth
                            (secondary to accuracy)
    QML_MAX_QUBITS          hard cap (only enforced when MINIMIZE_RESOURCES=1):
    QML_MAX_GATES           a circuit over ANY set cap is still scored but
    QML_MAX_DEPTH           marked ok=false "NOT VIABLE" so it cannot win.
                            Unset / 0 = no limit for that resource.

Refresh the seed baselines after changing dim/kernel/data:
    python qml_task.py baselines     # per-seed figures + paste-ready stats
    python qml_task.py classical     # classical-SVM references only (fast)
"""

from __future__ import annotations

import os
import sys
import datetime

# Honor .env even when THIS module is the entry point (e.g.
# `python qml_task.py baselines`) -- otherwise only run_discovery.py loads it
# and a direct run silently uses defaults. Must precede the env reads below.
# load_dotenv does NOT override already-set vars, so it is a no-op when
# run_discovery.py already called it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: save figures, never open a window
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score

from DiscoveryTask import DiscoveryTask
from eval_harness import evaluate_feature_map, load_dataset, SEED_FEATURE_MAPS

QML_DIM = int(os.environ.get("AGENT_DIM", "2"))
QML_KERNEL = os.environ.get("AGENT_KERNEL", "fidelity").strip().lower()
# 1 -> the generate/review prompts also push for fewer qubits/gates/lower depth
# as a SECONDARY objective (accuracy always stays primary)
QML_MINIMIZE_RESOURCES = os.environ.get("QML_MINIMIZE_RESOURCES", "0") != "0"


def _limit(name: str) -> int | None:
    """A hard resource cap from the env: unset / "" / "0" -> None (no limit)."""
    v = os.environ.get(name, "").strip()
    return int(v) if v and v != "0" else None


# HARD resource budgets, ONLY enforced when QML_MINIMIZE_RESOURCES is on. A
# circuit that busts any of these is scored but marked ok=False ("not viable"),
# so it cannot win. Set them in .env, e.g. QML_MAX_QUBITS=6.
QML_MAX_QUBITS = _limit("QML_MAX_QUBITS")
QML_MAX_GATES = _limit("QML_MAX_GATES")
QML_MAX_DEPTH = _limit("QML_MAX_DEPTH")

# dataset shape (kept fixed so baselines are reproducible)
TRAIN_SIZE = int(os.environ.get("QML_TRAIN_SIZE", "40"))
TEST_SIZE = int(os.environ.get("QML_TEST_SIZE", "10"))
DATA_SEED = int(os.environ.get("QML_DATA_SEED", "12345"))


def load_qml_data(dim: int = QML_DIM):
    """(Xtr, ytr, Xte, yte) for the configured dim / sizes / seed."""
    return load_dataset(dim, seed=DATA_SEED,
                        train_size=TRAIN_SIZE, test_size=TEST_SIZE)


# ---------------------------------------------------------------------------
# Config snapshot: a config.txt written into every folder that receives plots,
# recording every knob that affects a result -- so any figure is reproducible
# from the file next to it. (Mirrors clustering_task.write_plot_config.)
# ---------------------------------------------------------------------------
def pipeline_config(**extra) -> dict:
    """Every setting that changes a result, as an ordered dict."""
    cfg = {
        "AGENT_DIM": QML_DIM,
        "AGENT_KERNEL": QML_KERNEL,
        "QML_MINIMIZE_RESOURCES": QML_MINIMIZE_RESOURCES,
        # hard resource caps (None = unlimited; only enforced when minimize on)
        "QML_MAX_QUBITS": QML_MAX_QUBITS,
        "QML_MAX_GATES": QML_MAX_GATES,
        "QML_MAX_DEPTH": QML_MAX_DEPTH,
        "TRAIN_SIZE": TRAIN_SIZE,
        "TEST_SIZE": TEST_SIZE,
        "DATA_SEED": DATA_SEED,
    }
    cfg.update(extra)
    return cfg


def write_plot_config(plot_dir: str, **extra) -> str:
    """Write/refresh config.txt in plot_dir, snapshotting the settings behind
    the figures there."""
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "config.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# qml feature-map config -- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        for k, v in pipeline_config(**extra).items():
            f.write(f"{k} = {v}\n")
    return path


# ---------------------------------------------------------------------------
# Measured seed baselines: shown to the agent (in the seed library AND the
# generate system prompt) as "the number to beat". Values below are for the
# config in SEED_BASELINE_CONFIG; refresh with `python qml_task.py baselines`
# (the quantum-kernel eval is cheap, ~1s/seed) whenever dim/kernel/data change.
# ---------------------------------------------------------------------------
# NOTE: at dim=2/fidelity the zz_l2 seed already hits accuracy 1.0 (nothing to
# beat) and classical SVMs score ~0.15-0.25 (the ad_hoc set is classically
# hard by design). Higher dim (e.g. AGENT_DIM=4) leaves real headroom -- run
# `python qml_task.py baselines` to refresh for your dim/kernel.
# paste over SEED_BASELINE_STATS in qml_task.py:
SEED_BASELINE_CONFIG = "dim=3, kernel=dqc1, train=80, test=20, seed=12345"
SEED_BASELINE_STATS = {
    "zz_l2": {"accuracy": 0.675, "qubits": 3, "gates": 30, "depth": 22},
    "z_first_order": {"accuracy": 0.35, "qubits": 3, "gates": 6, "depth": 2},
    "reupload_ring": {"accuracy": 0.525, "qubits": 3, "gates": 27, "depth": 15},
}

def _annotated_seeds(seed_stats: dict | None) -> dict:
    """The seed library with each map's measured accuracy prepended as a
    comment, so view_seed_library shows code AND score together."""
    if not seed_stats:
        return dict(SEED_FEATURE_MAPS)
    out = {}
    for name, code in SEED_FEATURE_MAPS.items():
        s = seed_stats.get(name)
        if isinstance(s, dict) and "accuracy" in s:
            header = (f"# measured ({SEED_BASELINE_CONFIG}): "
                      f"accuracy={s['accuracy']:.3f}, "
                      f"qubits={s.get('qubits', '?')}, gates={s['gates']}, "
                      f"depth={s['depth']}")
        else:
            header = "# (no measured baseline yet)"
        out[name] = f"\n{header}\n{code.strip()}\n"
    return out


def save_seed_comparison_chart(stats: dict, path: str, note: str = "") -> str:
    """Horizontal bars of test accuracy per feature map (and any classical
    references / the agent, if present), sorted best on top; rows whose name
    starts with '>>' (the agent) are bolded. Error entries are skipped."""
    rows = [(n, s) for n, s in stats.items()
            if isinstance(s, dict) and "accuracy" in s]
    if not rows:
        return ""
    write_plot_config(os.path.dirname(path) or ".", chart_note=note)
    rows.sort(key=lambda kv: kv[1]["accuracy"])       # barh: last row on top
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
    ax.set_title("Feature-map comparison" + (f" — {note}" if note else ""),
                 fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compute_seed_baselines(dim: int = QML_DIM, kernel: str = QML_KERNEL,
                           plot_dir: str | None = "seed_baselines") -> dict:
    """Score every seed feature map with the real evaluator and return
    {name: {"accuracy", "gates", "depth"}}.

    Also prints the dict as paste-ready Python for SEED_BASELINE_STATS and
    (with plot_dir) saves one kernel/prediction figure per seed plus the
    comparison bar chart. Use whenever dim/kernel/data change.
    """
    data = load_qml_data(dim)
    config = (f"dim={dim}, kernel={kernel}, train={TRAIN_SIZE}, "
              f"test={TEST_SIZE}, seed={DATA_SEED}")
    stats: dict = {}
    for name, code in SEED_FEATURE_MAPS.items():
        m = evaluate_feature_map(code, data=data, kernel=kernel,
                                 plot=bool(plot_dir), iteration=f"seed_{name}",
                                 plot_dir=plot_dir or "plots")
        if m.get("ok"):
            stats[name] = {"accuracy": round(m["accuracy"], 4),
                           "qubits": m["n_qubits"], "gates": m["n_gates"],
                           "depth": m["depth"]}
            print(f"  {name:20s} accuracy={m['accuracy']:.4f} "
                  f"qubits={m['n_qubits']} gates={m['n_gates']} depth={m['depth']}")
        else:
            stats[name] = {"error": m.get("error")}
            print(f"  {name:20s} ERROR: {m.get('error')}")
    if plot_dir:
        chart = save_seed_comparison_chart(
            stats, os.path.join(plot_dir, "seed_comparison.png"), note=config)
        print(f"  [chart] {chart}")
    print("\n# paste over SEED_BASELINE_STATS in qml_task.py:")
    print(f'SEED_BASELINE_CONFIG = "{config}"')
    print("SEED_BASELINE_STATS = {")
    for n, s in stats.items():
        if "accuracy" in s:
            print(f'    "{n}": {{"accuracy": {s["accuracy"]}, '
                  f'"qubits": {s["qubits"]}, "gates": {s["gates"]}, '
                  f'"depth": {s["depth"]}}},')
    print("}")
    return stats


def classical_baselines(data=None, dim: int = QML_DIM,
                        plot_dir: str | None = None) -> dict:
    """Classical-kernel SVM references on the SAME train/test split: an RBF-SVM
    and a linear-SVM. This is the "is the quantum kernel beating classical?"
    yardstick -- the supervised analogue of Parzen / k-means for clustering.
    Returns {"svm_rbf": {"accuracy"}, "svm_linear": {"accuracy"}}.
    """
    if data is None:
        data = load_qml_data(dim)
    Xtr, ytr, Xte, yte = data
    out: dict = {}
    for name, kern in (("svm_rbf", "rbf"), ("svm_linear", "linear")):
        try:
            clf = svm.SVC(kernel=kern).fit(Xtr, ytr)
            acc = accuracy_score(clf.predict(Xte), yte)
            out[name] = {"accuracy": round(float(acc), 4)}
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    if plot_dir:
        write_plot_config(plot_dir, methods="RBF-SVM + linear-SVM")
    return out


def format_classical_report(res: dict) -> str:
    """classical_baselines() output as a printable table."""
    lines = []
    for name, r in res.items():
        if "accuracy" in r:
            lines.append(f"  {name:14s} accuracy={r['accuracy']:.3f}")
        else:
            lines.append(f"  {name:14s} ERROR: {r.get('error')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts, one per agent role
# ---------------------------------------------------------------------------
def _system_prompts(dim: int, seed_stats: dict | None = None,
                    minimize_resources: bool = False) -> dict[str, str]:
    # effective upper qubit bound stated in the contract: the enforced hard cap
    # when one is set (minimize on), else the soft dim+4 default
    q_hi = (QML_MAX_QUBITS if (minimize_resources and QML_MAX_QUBITS is not None)
            else dim + 4)
    # optional secondary objective: circuit economy (toggle, off by default)
    resource_block, resource_note = "", ""
    if minimize_resources:
        # hard, enforced budgets (only the ones actually set in .env)
        limits = []
        if QML_MAX_QUBITS is not None:
            limits.append(f"at most {QML_MAX_QUBITS} qubits")
        if QML_MAX_GATES is not None:
            limits.append(f"at most {QML_MAX_GATES} gates")
        if QML_MAX_DEPTH is not None:
            limits.append(f"depth at most {QML_MAX_DEPTH}")
        limit_line = ""
        if limits:
            limit_line = (
                "\nHARD RESOURCE LIMITS (ENFORCED): your circuit must use "
                + ", ".join(limits) + ". A circuit that exceeds ANY of these is "
                "still scored but marked NOT VIABLE (ok=false) no matter how "
                "accurate it is -- it cannot be submitted or win, and the "
                "evaluate result will say by how much it went over. Stay within "
                "budget; these limits override the qubit range in the contract "
                "above.")
        resource_block = (
            "\n\n# Secondary objective: circuit economy\n"
            "Among candidates with comparable accuracy, PREFER fewer qubits, "
            "fewer gates, and lower depth (all reported in every evaluation "
            "result). Never trade accuracy away for economy: beat the baseline "
            "first, then simplify -- drop gates that do not change the score, "
            "use the fewest qubits that work, and prefer shallow layers."
            + limit_line)
        resource_note = (" Also flag waste: qubits, gates, or depth that do "
                         "not earn their accuracy.")
    # "number to beat", built from the measured seed baselines (if provided)
    rows = [(n, s) for n, s in (seed_stats or {}).items()
            if isinstance(s, dict) and "accuracy" in s]
    beat_note, baseline_block = "", ""
    if rows:
        best_name, best_s = max(rows, key=lambda kv: kv[1]["accuracy"])
        table = "\n".join(f"  - {n}: accuracy={s['accuracy']:.3f}, "
                          f"qubits={s.get('qubits', '?')}, gates={s['gates']}, "
                          f"depth={s['depth']}" for n, s in rows)
        beat_note = (f" The measured best seed is {best_name} at accuracy "
                     f"{best_s['accuracy']:.3f}; the goal is to EXCEED it.")
        baseline_block = (
            f"\n\n# Seed baselines to beat ({SEED_BASELINE_CONFIG})\n{table}\n"
            f"NUMBER TO BEAT: {best_name} at accuracy {best_s['accuracy']:.3f} -- "
            f"success = accuracy strictly ABOVE it. First reason about why the "
            f"strong seeds win and weak ones fail, then extend/hybridize/rethink "
            f"them.\nSUBMISSION RULE: submit your OWN circuit, never a seed "
            f"verbatim (nor one with only cosmetic edits). Even if nothing beat "
            f"{best_name}, submit your best original and say it fell short -- "
            f"returning a seed is an automatic failure.")
    return {
        "explore": (
            f"You are a research assistant for designing quantum feature maps for a "
            f"{dim}-feature binary-classification quantum-kernel SVM. Use the lookup "
            f"tools to gather relevant encoding strategies, entanglement patterns, and "
            f"prior results. Take concise, actionable notes the designer can build on. "
            f"When you have enough, stop calling tools and summarize your notes."
            + beat_note
        ),
        "generate": (
            f"You are an expert quantum circuit designer. Design a quantum feature "
            f"map for a {dim}-feature binary classification task that MAXIMIZES SVM "
            f"test accuracy while keeping the gate count modest.\n"
            f"Strategy: (1) LOCAL SEARCH FIRST -- your first trial each session is "
            f"the best seed with ONE change (one rotation, one entangler, one "
            f"layer); only redesign after scoring that. (2) WATCH OVERFITTING / "
            f"KERNEL CONCENTRATION -- the test set is small and very deep/wide "
            f"circuits overfit or drive the Gram matrix near-constant (lost class "
            f"separation); if a bigger circuit scores WORSE, simplify.\n"
            f"Contract: define exactly one build_circuit(x)->QuantumCircuit; x is "
            f"a length-{dim} array; {dim}-{q_hi} qubits; no imports (`np`, "
            f"`QuantumCircuit` in scope), no measurements, trainable params, or "
            f"classical nonlinear preprocessing of x. Present each candidate as a "
            f"```python block, THEN call `evaluate` on it. Refine from the scores, "
            f"or reply WITHOUT a tool call to submit."
            + baseline_block + resource_block
        ),
        "review": (
            "You are a critical reviewer of quantum feature-map designs for a "
            "quantum-kernel SVM. Given a candidate and its measured accuracy/gate "
            "stats, give ONE specific, actionable improvement (entanglement pattern, "
            "data re-uploading, encoding rotations, or lower depth). Be concise "
            "but give as much detail as the improvement warrants, plus a brief "
            "note of any expert findings worth remembering. "
            "If it errored, state the concrete fix."
            + beat_note + resource_note
        ),
        "deploy": (
            "You are writing the final report for an automated quantum-circuit search. "
            "Given the best feature map and its stats, explain in a short paragraph "
            "what was designed, its final accuracy/gate stats, and the design rationale."
        ),
    }


def make_qml_task(dim: int = QML_DIM, kernel: str = QML_KERNEL,
                  seed_stats: dict | None = None,
                  minimize_resources: bool = QML_MINIMIZE_RESOURCES,
                  ) -> DiscoveryTask:
    """Build the supervised feature-map DiscoveryTask.

    seed_stats: measured per-seed baselines shown to the agent as the numbers
    to beat. Default: the hardcoded SEED_BASELINE_STATS (valid for
    SEED_BASELINE_CONFIG -- refresh if dim/kernel/data changed); pass
    compute_seed_baselines(...) output for fresh values, or {} to disable.
    minimize_resources: adds the optional circuit-economy prompt section
    (env toggle: QML_MINIMIZE_RESOURCES=1).
    """
    if seed_stats is None:
        seed_stats = SEED_BASELINE_STATS
    data = load_qml_data(dim)

    def _evaluate(code: str, *args, **kwargs) -> dict:
        # figures live in the task's per-run folder, next to the transcript;
        # plots/ subfolder mirrors clustering_task.py
        plot_dir = kwargs.setdefault("plot_dir", os.path.join(task.run_dir, "plots"))
        # enforce the hard resource budget ONLY when minimize_resources is on
        if minimize_resources:
            kwargs.setdefault("max_qubits", QML_MAX_QUBITS)
            kwargs.setdefault("max_gates", QML_MAX_GATES)
            kwargs.setdefault("max_depth", QML_MAX_DEPTH)
        m = evaluate_feature_map(code, data=data, kernel=kernel, *args, **kwargs)
        if kwargs.get("plot"):
            write_plot_config(plot_dir)    # config.txt next to the kernel figures
        return m

    task = DiscoveryTask(
        name=f"qml_feature_map_dim{dim}_{kernel}",
        system_prompts=_system_prompts(dim, seed_stats, minimize_resources),
        seeds={"feature_map": _annotated_seeds(seed_stats)},
        evaluate_fn=_evaluate,
        # placeholders -- wire real RAG here later (see AgentPipeline plan)
        knowledge_sources=["arXiv:2210.09275", "arXiv:1804.11326 (Havlicek)"],
        documentation_source="qiskit (QuantumCircuit API)",
    )
    # record the resource-limit state in the run transcript (also snapshotted
    # into each config.txt via pipeline_config)
    if minimize_resources:
        task.log("RESOURCE LIMITS (enforced)",
                 f"max_qubits={QML_MAX_QUBITS}  max_gates={QML_MAX_GATES}  "
                 f"max_depth={QML_MAX_DEPTH}   (None = no limit; an over-budget "
                 f"circuit is still scored but marked ok=false so it cannot win)")
    else:
        task.log("RESOURCE LIMITS", "QML_MINIMIZE_RESOURCES off -- no caps enforced")
    return task


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "baselines":
        # refresh the seed baselines (per-seed kernel figures + comparison
        # chart + paste-ready stats), THEN the classical-SVM references --
        # everything into one timestamped seed_baselines/<stamp>/ folder
        run_dir = os.path.join("seed_baselines",
                               datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        compute_seed_baselines(plot_dir=run_dir)
        print("\nClassical baselines (same data):")
        print(format_classical_report(classical_baselines(plot_dir=run_dir)))
    elif len(sys.argv) > 1 and sys.argv[1] == "classical":
        run_dir = os.path.join("seed_baselines",
                               datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(format_classical_report(classical_baselines(plot_dir=run_dir)))
    else:
        task = make_qml_task()
        print(task.describe())
        seed = next(iter(task.seeds("feature_map").values()))
        print("baseline eval:", task.evaluate(seed, plot=True))
