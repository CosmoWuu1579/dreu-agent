"""
Evaluation harness for agent-designed quantum classification HEADS
(the stronglyentangled/ sibling of eval_harness.py, but styled after the paper's
own source: CNN-Q*.py from github.com/VeraaaCUI/StronglyEntangled).

The agent proposes a quantum head create_qnn(n) -> EstimatorQNN. We drop it into
the SAME hybrid tail the paper uses -- Linear(features -> n) -> TorchConnector(qnn)
-> Linear(1 -> 1) -> cat((x, 1-x)) -- and train it with NAdam + CrossEntropyLoss,
exactly like the source. The ONLY change from the paper is that the 4-conv CNN
backbone is replaced by precomputed features (a frozen-CNN proxy): each training
example is already the small feature vector the CNN would emit, so only the head
is trained. Swap `load_se_data` for real Br35H CNN features later and the rest is
unchanged.

Kept deliberately parallel to the paper's code so the two are easy to compare:
same imports, same Net.forward tail, same NAdam / CrossEntropyLoss training loop.
"""

from __future__ import annotations

import os
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: save figures, never open a window
import matplotlib.pyplot as plt

# --- torch stack (as in the paper's source) ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch import cat, no_grad, manual_seed

# --- qiskit stack (as in the paper's source) ---
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
# functional constructors (non-deprecated) -- the class forms ZFeatureMap /
# ZZFeatureMap / RealAmplitudes / EfficientSU2 are deprecated in Qiskit 2.1 and
# removed in 3.0, so we build with the functions and only KEEP the classes in
# scope as a backward-compat fallback (imported defensively below).
from qiskit.circuit.library import (z_feature_map, zz_feature_map,
                                     real_amplitudes, efficient_su2)
try:
    from qiskit.circuit.library import (ZFeatureMap, ZZFeatureMap,
                                         RealAmplitudes, EfficientSU2)
    _LEGACY_LIB = {"ZFeatureMap": ZFeatureMap, "ZZFeatureMap": ZZFeatureMap,
                   "RealAmplitudes": RealAmplitudes, "EfficientSU2": EfficientSU2}
except ImportError:                          # Qiskit 3.0+: classes gone
    _LEGACY_LIB = {}
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.utils import algorithm_globals

from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Evaluator-level config (data + training). Policy caps (max_qubits, ...) are
# passed in by the task, so this module stays policy-free like eval_harness.py.
# ---------------------------------------------------------------------------
SE_DIM = int(os.environ.get("AGENT_DIM", "2"))           # features / qubits
SE_TRAIN_SIZE = int(os.environ.get("SE_TRAIN_SIZE", "20"))
SE_TEST_SIZE = int(os.environ.get("SE_TEST_SIZE", "10"))
SE_EPOCHS = int(os.environ.get("SE_EPOCHS", "40"))       # NAdam epochs/candidate
SE_LR = float(os.environ.get("SE_LR", "0.05"))
SE_DATA_SEED = int(os.environ.get("SE_DATA_SEED", "12345"))
SE_EVAL_EVERY = int(os.environ.get("SE_EVAL_EVERY", "1"))  # validate every k epochs for the log


# ---------------------------------------------------------------------------
# Dataset = the compressed features the frozen CNN would hand to the quantum
# head. Stand-in: the paper's ad_hoc set (n in {2,3}); synthetic for n >= 4.
# REAL Br35H LATER: replace this function's body with "load Br35H, run the
# frozen CNN, return its feature-layer outputs + labels". Nothing else changes.
# ---------------------------------------------------------------------------
def load_se_data(n: int = SE_DIM, seed: int = SE_DATA_SEED,
                 train_size: int = SE_TRAIN_SIZE, test_size: int = SE_TEST_SIZE):
    """(Xtr, ytr, Xte, yte); X rows are feature vectors, y in {0,1}."""
    if n <= 3:
        algorithm_globals.random_seed = seed
        Xtr, ytr, Xte, yte = ad_hoc_data(
            training_size=train_size, test_size=test_size, n=n, gap=0.3,
            plot_data=False, one_hot=False,
        )
        return Xtr, ytr.astype(int), Xte, yte.astype(int)
    X, y = make_classification(
        n_samples=train_size + test_size, n_features=n, n_informative=n,
        n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=seed,
    )
    X = MinMaxScaler((0, 2 * np.pi)).fit_transform(X)
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]


# ---------------------------------------------------------------------------
# Turn agent code (a string) into an EstimatorQNN via create_qnn(n)
# ---------------------------------------------------------------------------
def compile_qnn(code: str, n: int) -> EstimatorQNN:
    """Exec agent code in a restricted namespace and return create_qnn(n).

    The code MUST define create_qnn(n) -> EstimatorQNN, with `input_params` set
    to the n encoder inputs, `weight_params` to the trainable weights, a SINGLE
    observable (scalar output), and input_gradients=True (so gradients flow back
    into the classical fc2 through the TorchConnector). No imports: the quantum
    building blocks below are already in scope.
    """
    if "import" in code:
        raise ValueError("imports are not allowed in the create_qnn code")
    ns: dict = {
        "np": np, "numpy": np,
        "QuantumCircuit": QuantumCircuit, "ParameterVector": ParameterVector,
        "z_feature_map": z_feature_map, "zz_feature_map": zz_feature_map,
        "real_amplitudes": real_amplitudes, "efficient_su2": efficient_su2,
        "SparsePauliOp": SparsePauliOp, "EstimatorQNN": EstimatorQNN,
    }
    ns.update(_LEGACY_LIB)   # keep deprecated class names working if importable
    exec(code, ns)
    if "create_qnn" not in ns or not callable(ns["create_qnn"]):
        raise ValueError("code must define a callable create_qnn(n)")
    qnn = ns["create_qnn"](n)
    if not isinstance(qnn, EstimatorQNN):
        raise ValueError("create_qnn(n) must return an EstimatorQNN")
    # ensure gradients reach the classical fc2, even if the agent forgot the flag
    try:
        qnn.input_gradients = True
    except Exception:
        pass
    return qnn


# ---------------------------------------------------------------------------
# The hybrid net -- this is the paper's `Net` with the 4-conv backbone removed
# (its 8x8x128 -> 512 output is replaced by the precomputed feature vector).
# The tail (fc2 -> QNN -> fc3 -> cat) and the forward code are copied from the
# source so the two line up gate-for-gate.
# ---------------------------------------------------------------------------
class HybridQCNN(nn.Module):
    def __init__(self, in_dim, qnn):
        super().__init__()
        # source maps 512 CNN features -> n qubits; here in_dim = feature width
        self.fc2 = nn.Linear(in_dim, qnn.num_inputs)
        self.qnn = TorchConnector(qnn)         # the trainable quantum head
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc2(x)
        x = self.qnn(x)
        x = self.fc3(x)
        return cat((x, 1 - x), -1)             # 2 logits, exactly as the source


# ---------------------------------------------------------------------------
# Plot: test-set predictions on the first two feature dims + accuracy
# ---------------------------------------------------------------------------
def _save_prediction_image(Xte, y_pred, yte, acc, iteration, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_iter{iteration}" if iteration is not None else ""
    path = os.path.join(plot_dir, f"qcnn_{stamp}{tag}.png")
    Xte = np.asarray(Xte); y_pred = np.asarray(y_pred); yte = np.asarray(yte)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(Xte[y_pred == 0, 0], Xte[y_pred == 0, 1], marker="s",
               facecolors="none", edgecolors="b", label="pred 0")
    ax.scatter(Xte[y_pred == 1, 0], Xte[y_pred == 1, 1], marker="o",
               facecolors="none", edgecolors="r", label="pred 1")
    wrong = y_pred != yte
    if wrong.any():
        ax.scatter(Xte[wrong, 0], Xte[wrong, 1], marker="x", c="k", s=80,
                   label="wrong")
    ax.set_title(f"QCNN head  |  accuracy = {acc:.3f}"
                 + (f"  |  iteration {iteration}" if iteration is not None else ""))
    ax.set_xlabel("feature[0]"); ax.set_ylabel("feature[1]")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# The evaluator: compile create_qnn -> train the hybrid head (NAdam / CE) -> score.
# Returns metrics or an 'error' field (this is the "tool" the agent calls).
# ---------------------------------------------------------------------------
def evaluate_qcnn(code: str, data=None, n: int = SE_DIM, epochs: int = SE_EPOCHS,
                  lr: float = SE_LR, seed: int = SE_DATA_SEED, plot: bool = False,
                  iteration=None, plot_dir: str = "plots",
                  max_qubits: int | None = None, max_gates: int | None = None,
                  max_depth: int | None = None,
                  max_weights: int | None = None) -> dict:
    """Compile the quantum head, wrap it in the paper's fc2->QNN->fc3 net, train
    with NAdam + CrossEntropyLoss on the fixed features, score on the test set.

    Returns accuracy (primary), f1, and the circuit's cost (n_qubits / n_gates /
    depth / n_weights, where n_weights = trainable QNN params, the paper's #Para).

    max_*: HARD budgets (None = off). The head is STILL trained and scored, but
    if it busts any set limit the result is marked ok=False ("not viable").
    """
    try:
        qnn = compile_qnn(code, n)
        Xtr, ytr, Xte, yte = data if data is not None else load_se_data(n)
        Xtr = np.asarray(Xtr, dtype=np.float32); Xte = np.asarray(Xte, dtype=np.float32)
        ytr = np.asarray(ytr, dtype=int); yte = np.asarray(yte, dtype=int)

        if qnn.num_inputs != n:
            raise ValueError(f"create_qnn(n) must take exactly n={n} feature "
                             f"inputs (input_params); got num_inputs={qnn.num_inputs}")
        if qnn.num_weights < 1:
            raise ValueError("the quantum head has no trainable weights "
                             "(weight_params is empty) -- add a variational ansatz")
        if tuple(np.atleast_1d(qnn.output_shape)) != (1,):
            raise ValueError("use exactly ONE observable so the head outputs a "
                             f"single score; got output_shape={qnn.output_shape}")

        qc = qnn.circuit
        # decompose so library blocks (ZZFeatureMap / RealAmplitudes / ...) are
        # unrolled into primitive gates -- otherwise count_ops sees each block as
        # a single instruction and reports gates=2 / depth=2 for every head.
        qc_flat = qc.decompose(reps=3)
        n_qubits = qc.num_qubits
        n_gates = int(sum(qc_flat.count_ops().values()))
        depth = qc_flat.depth()
        n_weights = int(qnn.num_weights)

        # --- train the hybrid head: NAdam + CrossEntropyLoss (paper's recipe) ---
        manual_seed(seed)
        model = HybridQCNN(Xtr.shape[1], qnn).to(device)
        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).long().to(device)
        Xte_t = torch.from_numpy(Xte).to(device)
        optimizer = optim.NAdam(model.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        # per-epoch progress log, mirroring the paper's training_performance_*.txt
        # (epoch, train loss, validation accuracy) -- printed AND appended to a txt
        # in plot_dir. Validation runs every SE_EVAL_EVERY epochs (and on the last).
        os.makedirs(plot_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_iter{iteration}" if iteration is not None else ""
        log_path = os.path.join(plot_dir, f"training_qcnn_{stamp}{tag}.txt")
        with open(log_path, "w", encoding="utf-8") as flog:
            flog.write("epoch, train_loss, val_accuracy(%)\n")

        def _val_predict():
            model.eval()
            with no_grad():
                return model(Xte_t).argmax(dim=1).cpu().numpy()

        model.train()
        last_loss = float("nan")
        y_pred = None
        for epoch in range(epochs):                  # full-batch: the data is small
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(Xtr_t)
            loss = loss_func(output, ytr_t)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

            if epoch % SE_EVAL_EVERY == 0 or epoch == epochs - 1:
                y_pred = _val_predict()
                val_str = f"{100.0 * accuracy_score(yte, y_pred):.2f}%"
            else:
                val_str = "-"
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(f"{epoch + 1}, {last_loss:.4f}, {val_str}\n")
            print(f"Epoch: {epoch + 1}/{epochs}, Training Loss: {last_loss:.4f}, "
                  f"Validation Accuracy: {val_str}")

        # final predictions from the last evaluation (last epoch always evaluates;
        # guard the epochs==0 edge)
        if y_pred is None:
            y_pred = _val_predict()

        acc = accuracy_score(yte, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yte, y_pred, average="binary", zero_division=0)
        n_params_total = int(sum(p.numel() for p in model.parameters()))

        metrics = {
            "ok": True,
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "train_loss": round(last_loss, 4),
            "n_qubits": n_qubits,
            "n_gates": n_gates,
            "depth": depth,
            "n_weights": n_weights,            # trainable QNN params (paper #Para)
            "n_params_total": n_params_total,  # incl. classical fc2 + fc3
            "train_log": log_path,
        }
        if plot:
            metrics["plot_path"] = _save_prediction_image(
                Xte, y_pred, yte, acc, iteration, plot_dir)

        # HARD resource budget: score is real, but bust a cap -> not viable.
        violations = []
        if max_qubits is not None and n_qubits > max_qubits:
            violations.append(f"qubits {n_qubits} > limit {max_qubits}")
        if max_gates is not None and n_gates > max_gates:
            violations.append(f"gates {n_gates} > limit {max_gates}")
        if max_depth is not None and depth > max_depth:
            violations.append(f"depth {depth} > limit {max_depth}")
        if max_weights is not None and n_weights > max_weights:
            violations.append(f"weights {n_weights} > limit {max_weights}")
        if violations:
            metrics["ok"] = False
            metrics["resource_violation"] = violations
            metrics["error"] = (
                "NOT VIABLE -- head exceeds the resource budget: "
                + "; ".join(violations)
                + f". It scored accuracy={metrics['accuracy']:.3f}, but that does "
                "NOT count while over budget and this head cannot be submitted. "
                "Reduce qubits/gates/depth/weights and try a smaller circuit.")
        return metrics
    except Exception as exc:  # returned to the agent for self-correction
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def classical_baselines(data=None, dim: int = SE_DIM) -> dict:
    """Classical-kernel SVM references (RBF + linear) on the SAME features --
    the "is the quantum head beating a plain SVM?" yardstick."""
    if data is None:
        data = load_se_data(dim)
    Xtr, ytr, Xte, yte = data
    out: dict = {}
    for name, kern in (("svm_rbf", "rbf"), ("svm_linear", "linear")):
        try:
            clf = svm.SVC(kernel=kern).fit(Xtr, ytr)
            out[name] = {"accuracy": round(float(accuracy_score(clf.predict(Xte), yte)), 4)}
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def format_classical_report(res: dict) -> str:
    lines = []
    for name, r in res.items():
        if "accuracy" in r:
            lines.append(f"  {name:12s} accuracy={r['accuracy']:.3f}")
        else:
            lines.append(f"  {name:12s} ERROR: {r.get('error')}")
    return "\n".join(lines)
