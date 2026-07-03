"""
Evaluation harness for agent-designed quantum feature maps.

This is the "tool" the LangGraph agent calls (analogous to Knipfer et al.'s
TrainCustomSimpleQNNTool and Sakka et al.'s Evaluation component). The agent
proposes a feature map u(x) as a string of Python code; this module executes it,
builds a fidelity quantum kernel via statevector simulation, trains a classical
SVM, and returns performance metrics.

Design choice: the agent designs the *bare* feature map (no control qubit). We
score it with the fidelity kernel
        K(x, x') = |<psi(x')|psi(x)>|^2,   psi(x) = u(x)|0...0>
exactly as in both reference papers. Unlike the DQC1 kernel (code.py) this one
does NOT exponentially concentrate, so it scales to higher dimensions.

Dimensionality: everything keys off len(x) = number of features = number of
qubits the feature map uses. Use load_dataset(n) to get an n-feature dataset.

GPU: statevector simulation runs on GPU automatically when qiskit-aer exposes a
'GPU' device (requires qiskit-aer-gpu, i.e. CUDA on Linux/WSL2); otherwise it
falls back to the fast CPU path. Force it with the QISKIT_DEVICE env var
('auto' | 'CPU' | 'GPU'). At <= ~18 qubits GPU gives no speedup.
"""

from __future__ import annotations

import os
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: save figures to disk, never open a window
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, Operator

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.utils import algorithm_globals


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def load_adhoc(seed: int = 12345, n: int = 2, train_size: int = 20, test_size: int = 5):
    """ad_hoc dataset with n features (ad_hoc_data supports n in {2, 3}).

    train_size / test_size are per-class sample counts (ad_hoc_data's convention),
    so the returned sets hold 2*train_size / 2*test_size points total.
    """
    algorithm_globals.random_seed = seed
    Xtr, ytr, Xte, yte, _ = ad_hoc_data(
        training_size=train_size, test_size=test_size, n=n, gap=0.3,
        plot_data=False, one_hot=False, include_sample_total=True,
    )
    return Xtr, ytr, Xte, yte


def load_dataset(n: int = 2, seed: int = 12345, train_size: int = 20, test_size: int = 5):
    """n-feature binary dataset scaled to [0, 2*pi] (gate-angle range).

    Uses the paper's ad_hoc set for n in {2, 3}; for n >= 4 (where ad_hoc is
    unsupported) it falls back to a synthetic sklearn dataset.

    train_size / test_size set the number of training / test points. For the
    ad_hoc path they are per-class counts (see load_adhoc); for the synthetic
    fallback they are the total number of points in each split.
    """
    if n <= 3:
        return load_adhoc(seed, n, train_size, test_size)
    X, y = make_classification(
        n_samples=train_size + test_size, n_features=n, n_informative=n,
        n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=seed,
    )
    X = MinMaxScaler((0, 2 * np.pi)).fit_transform(X)
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]


# ---------------------------------------------------------------------------
# GPU-aware statevector simulation
# ---------------------------------------------------------------------------
_GPU_CACHE = None


def _gpu_available() -> bool:
    global _GPU_CACHE
    if _GPU_CACHE is None:
        try:
            from qiskit_aer import AerSimulator
            _GPU_CACHE = "GPU" in AerSimulator().available_devices()
        except Exception:
            _GPU_CACHE = False
    return _GPU_CACHE


def resolve_device(device: str | None = None) -> str:
    device = device or os.environ.get("QISKIT_DEVICE", "auto")
    if device == "auto":
        return "GPU" if _gpu_available() else "CPU"
    return device


def statevectors(circuits, device: str | None = None):
    """Return the statevector (numpy array) of each circuit."""
    device = resolve_device(device)
    if device == "GPU":
        from qiskit_aer import AerSimulator
        backend = AerSimulator(method="statevector", device="GPU")
        circs = []
        for qc in circuits:
            c = qc.copy()
            c.save_statevector()
            circs.append(transpile(c, backend))
        result = backend.run(circs).result()
        return [np.asarray(result.get_statevector(i)) for i in range(len(circs))]
    # CPU: quantum_info is faster than Aer for the small circuits here
    return [np.asarray(Statevector(qc)) for qc in circuits]


# ---------------------------------------------------------------------------
# DQC1 (paper) kernel: wraps ANY feature map u(x) in the Hadamard test of
# code.py, so an agent-designed feature map can be reproduced in the exact
# setting of arXiv:2210.09275. u = build_circuit(x_i) . build_circuit(x_j)^dag
# is *controlled* by a clean qubit acting on maximally-mixed targets, and the
# kernel value is 2*|rho_01| of that control qubit. (Concentrates with n --
# useful for reproduction at n=2, not as a search objective.)
# ---------------------------------------------------------------------------
def dqc1_kernel_value(build_circuit, x_i, x_j) -> float:
    u_i = build_circuit(x_i)
    n = u_i.num_qubits
    U = u_i.compose(build_circuit(x_j).inverse())      # u(x_i) u(x_j)^dagger
    controlled_U = U.to_gate().control(1)
    qc = QuantumCircuit(2 * n + 1)
    qc.h(0)                                             # control qubit -> |+>
    for t in range(1, n + 1):                          # targets -> maximally mixed
        qc.h(t)
        qc.cx(t, t + n)
    qc.append(controlled_U, [0] + list(range(1, n + 1)))
    rho_c = partial_trace(DensityMatrix(qc), list(range(1, 2 * n + 1)))
    return 2 * abs(rho_c.data[0, 1])


# ---------------------------------------------------------------------------
# Turn agent code (a string) into a callable build_circuit(x) -> QuantumCircuit
# ---------------------------------------------------------------------------
def compile_feature_map(code: str):
    """Exec agent code in a restricted namespace and return build_circuit.

    The code MUST define build_circuit(x) -> QuantumCircuit (no measurements).
    Only numpy and QuantumCircuit are in scope (no imports allowed).
    """
    if "import" in code:
        raise ValueError("imports are not allowed in the feature-map code")
    ns: dict = {"np": np, "numpy": np, "QuantumCircuit": QuantumCircuit}
    exec(code, ns)
    if "build_circuit" not in ns or not callable(ns["build_circuit"]):
        raise ValueError("code must define a callable build_circuit(x)")
    return ns["build_circuit"]


# ---------------------------------------------------------------------------
# Fidelity kernel + SVM evaluation  (this is what "scores the LLM's output")
# ---------------------------------------------------------------------------
def _save_kernel_image(Ktr, Xte, y_pred, yte, acc, kernel, iteration, plot_dir):
    """Save a code.py-style figure: kernel matrix + test-prediction scatter."""
    os.makedirs(plot_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_iter{iteration}" if iteration is not None else ""
    path = os.path.join(plot_dir, f"kernel_{stamp}{tag}.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    im = ax1.imshow(Ktr, origin="upper", interpolation="nearest")
    ax1.set_title("Training kernel matrix")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    Xte = np.asarray(Xte)
    y_pred, yte = np.asarray(y_pred), np.asarray(yte)
    ax2.scatter(Xte[y_pred == 0, 0], Xte[y_pred == 0, 1], marker="s",
                facecolors="none", edgecolors="b", label="pred 0")
    ax2.scatter(Xte[y_pred == 1, 0], Xte[y_pred == 1, 1], marker="o",
                facecolors="none", edgecolors="r", label="pred 1")
    wrong = y_pred != yte
    if wrong.any():
        ax2.scatter(Xte[wrong, 0], Xte[wrong, 1], marker="x", c="k", s=80, label="wrong")
    ax2.set_title("Test predictions (first 2 dims)")
    ax2.set_xlabel("x[0]"); ax2.set_ylabel("x[1]")
    ax2.legend(fontsize=7, loc="best")

    title = f"{kernel} kernel  |  accuracy = {acc:.3f}"
    if iteration is not None:
        title += f"  |  iteration {iteration}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate_feature_map(code: str, data=None, device: str | None = None,
                         kernel: str = "fidelity", plot: bool = False,
                         iteration=None, plot_dir: str = "plots") -> dict:
    """Compile, simulate, score. Returns metrics or an 'error' field on failure.

    kernel="fidelity" (default): scalable fidelity kernel -- the agent's search
        objective. kernel="dqc1": the paper's DQC1 kernel via the Hadamard test
        (reproduces arXiv:2210.09275; slower, concentrates with n).
    plot=True: save a kernel-matrix + prediction figure to
        plot_dir/kernel_<date>[_iterN].png and return its path under 'plot_path'.
    """
    try:
        build_circuit = compile_feature_map(code)
        Xtr, ytr, Xte, yte = data if data is not None else load_dataset()

        probe = build_circuit(Xtr[0])                     # dry run / validation
        if probe.num_clbits:
            raise ValueError("circuit must not contain measurements")

        if kernel == "fidelity":
            S_tr = np.array(statevectors([build_circuit(x) for x in Xtr], device))
            S_te = np.array(statevectors([build_circuit(x) for x in Xte], device))
            # fidelity Gram: K(a, b) = |<a|b>|^2  (batched inner products)
            Ktr = np.abs(S_tr.conj() @ S_tr.T) ** 2
            Kte = np.abs(S_te.conj() @ S_tr.T) ** 2
        elif kernel == "dqc1":
            # K(x, x') = |tr(u(x) u(x')^dag)| / 2^n = |<u(x'), u(x)>_HS| / 2^n.
            # Vectorized: precompute each unitary once, then one batched matmul --
            # far faster than building N^2 Hadamard-test density matrices.
            n = probe.num_qubits
            U_tr = np.array([Operator(build_circuit(x)).data.ravel() for x in Xtr])
            U_te = np.array([Operator(build_circuit(x)).data.ravel() for x in Xte])
            Ktr = np.abs(U_tr @ U_tr.conj().T) / (2 ** n)
            Kte = np.abs(U_te @ U_tr.conj().T) / (2 ** n)
        else:
            raise ValueError(f"unknown kernel: {kernel!r} (use 'fidelity' or 'dqc1')")

        clf = svm.SVC(kernel="precomputed").fit(Ktr, ytr)
        y_pred = clf.predict(Kte)
        acc = accuracy_score(y_pred, yte)

        metrics = {
            "ok": True,
            "accuracy": float(acc),
            "n_qubits": probe.num_qubits,
            "n_gates": int(sum(probe.count_ops().values())),
            "depth": probe.depth(),
            "kernel": kernel,
            "device": resolve_device(device) if kernel == "fidelity" else "CPU",
        }
        if plot:
            metrics["plot_path"] = _save_kernel_image(
                Ktr, Xte, y_pred, yte, acc, kernel, iteration, plot_dir)
        return metrics
    except Exception as exc:  # returned to the agent for self-correction
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Seed feature maps: a library the agent can start from / draw ideas from.
# Add new ones by dropping another entry into SEED_FEATURE_MAPS -- each value is
# a string defining build_circuit(x) under the same contract as agent code.
# ---------------------------------------------------------------------------
SEED_FEATURE_MAPS = {
    # The paper's ZZ-style l=2 map (first + second order, full entanglement).
    "zz_l2": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(2):                     # l = 2 layers
        for i in range(n):
            qc.h(i)
            qc.rz(2 * x[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.rz(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc
""",
    # First-order only (Z feature map): cheap baseline, no entanglement.
    "z_first_order": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        qc.rz(2 * x[i], i)
    return qc
""",
    # Data re-uploading with a linear CNOT chain and RY/RZ rotations.
    "reupload_ring": """
def build_circuit(x):
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(3):                     # 3 re-uploading blocks
        for i in range(n):
            qc.ry(2 * x[i], i)
            qc.rz(2 * x[i], i)
        for i in range(n):
            qc.cx(i, (i + 1) % n)          # ring of CNOTs
    return qc
""",
}

# Backwards-compatible default seed.
SEED_FEATURE_MAP = SEED_FEATURE_MAPS["zz_l2"]


def seed_library_text() -> str:
    """Render the seed library as a prompt-ready block."""
    blocks = []
    for name, code in SEED_FEATURE_MAPS.items():
        blocks.append(f"### {name}\n```python\n{code.strip()}\n```")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    for name, code in SEED_FEATURE_MAPS.items():
        print(f"[{name}]", evaluate_feature_map(code, data=load_dataset(2)))
