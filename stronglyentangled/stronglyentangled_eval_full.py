"""
FAITHFUL end-to-end evaluator: the paper's FULL hybrid network -- the 4-conv CNN
backbone AND the quantum head, trained TOGETHER (CNN weights NOT frozen), exactly
as in Cui & Huang's source (CNN-Q*.py) and Algorithm 1. This is the honest
counterpart to stronglyentangled_eval.py (which freezes the CNN and trains only
the head on cached features).

Same agent contract (create_qnn(n) -> EstimatorQNN) and same metrics dict, so it
is a DROP-IN alternative evaluator -- enable it for the discovery loop with
SE_FULL=1 (see stronglyentangled_task.make_stronglyentangled_task). It needs the
Br35H image dataset on disk and, realistically, a GPU.

Dataset layout (as the paper's code expects), pointed to by SE_DATA_DIR:
    <SE_DATA_DIR>/train/{yes,no}/*.jpg|*.png
    <SE_DATA_DIR>/test/{yes,no}/...
    <SE_DATA_DIR>/val/{yes,no}/...        (optional; falls back to test)

Speed knobs (trade faithfulness for time):
    SE_FULL_EPOCHS   training epochs        (default 300, the paper's value)
    SE_FULL_BATCH    batch size             (default 32)
    SE_FULL_LR       NAdam learning rate    (default 1e-5, the paper's Q*E2 value)
    SE_FULL_LIMIT    cap images PER SPLIT for a quick smoke run (0 = use all)
    SE_FULL_EVAL_EVERY  validate every k epochs for the progress log/print
                        (default 1 = every epoch, as in the paper)
"""

from __future__ import annotations

import os
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cat, no_grad, manual_seed
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from qiskit_machine_learning.connectors import TorchConnector

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# reuse the exact create_qnn contract / compilation + default dim from the
# frozen evaluator (no duplication, no circular import: that module is light)
from stronglyentangled_eval import compile_qnn, SE_DIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SE_DATA_DIR = os.environ.get("SE_DATA_DIR", "./data/Br35H")
SE_FULL_EPOCHS = int(os.environ.get("SE_FULL_EPOCHS", "300"))
SE_FULL_BATCH = int(os.environ.get("SE_FULL_BATCH", "32"))
SE_FULL_LR = float(os.environ.get("SE_FULL_LR", "0.00001"))
SE_FULL_LIMIT = int(os.environ.get("SE_FULL_LIMIT", "0"))     # 0 = all images
SE_FULL_EVAL_EVERY = int(os.environ.get("SE_FULL_EVAL_EVERY", "1"))  # validate every k epochs
SE_IMG = 128


# ---------------------------------------------------------------------------
# Dataset + transform: copied from the paper's source so pixels are identical
# ---------------------------------------------------------------------------
class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit: int = 0):
        self.transform = transform
        self.images, self.labels = [], []
        for sub, label in (("yes", 1), ("no", 0)):
            d = os.path.join(root_dir, sub)
            if not os.path.isdir(d):
                continue
            names = [f for f in os.listdir(d)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if limit:
                names = names[:limit]
            for f in names:
                self.images.append(os.path.join(d, f))
                self.labels.append(label)
        if not self.images:
            raise FileNotFoundError(
                f"no images under {root_dir!r} (expected yes/ and no/ subfolders). "
                f"Set SE_DATA_DIR to your Br35H split; see this file's docstring.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


transform = transforms.Compose([
    transforms.Resize(130),
    transforms.CenterCrop(SE_IMG),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

_LOADER_CACHE: dict = {}


def load_br35h(data_dir: str = SE_DATA_DIR, batch: int = SE_FULL_BATCH,
               limit: int = SE_FULL_LIMIT):
    """(train_loader, eval_loader); eval uses the val split, else test. Cached."""
    key = (data_dir, batch, limit)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]
    train = BrainDataset(os.path.join(data_dir, "train"), transform, limit)
    eval_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(data_dir, "test")
    ev = BrainDataset(eval_dir, transform, limit)
    loaders = (DataLoader(train, batch_size=batch, shuffle=True),
               DataLoader(ev, batch_size=batch, shuffle=False))
    _LOADER_CACHE[key] = loaders
    return loaders


# ---------------------------------------------------------------------------
# The paper's `Net`, verbatim structure, with the agent's qnn as the head.
# (fc2 512->n, TorchConnector(qnn), fc3 1->1, then cat((x, 1-x)).)
# ---------------------------------------------------------------------------
class FullHybridQCNN(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)     # 128x128 -> 8x8 after 4 pools
        self.fc2 = nn.Linear(512, qnn.num_inputs)  # -> n encoder inputs
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)
        x = self.fc3(x)
        return cat((x, 1 - x), -1)


def _save_curve(loss_hist, acc, iteration, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_iter{iteration}" if iteration is not None else ""
    path = os.path.join(plot_dir, f"qcnn_full_{stamp}{tag}.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(loss_hist)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss")
    ax.set_title(f"end-to-end training  |  val acc = {acc:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# The evaluator: compile create_qnn -> train the WHOLE net end-to-end -> score.
# Same signature + metrics dict as stronglyentangled_eval.evaluate_qcnn, so the
# task can swap it in unchanged. `data` is ignored (images are loaded here).
# ---------------------------------------------------------------------------
def evaluate_qcnn_full(code: str, data=None, n: int = SE_DIM,
                       epochs: int = SE_FULL_EPOCHS, lr: float = SE_FULL_LR,
                       seed: int = 12345, plot: bool = False, iteration=None,
                       plot_dir: str = "plots", max_qubits: int | None = None,
                       max_gates: int | None = None, max_depth: int | None = None,
                       max_weights: int | None = None) -> dict:
    """Train the full CNN+quantum hybrid end-to-end on Br35H, score on val/test."""
    try:
        qnn = compile_qnn(code, n)
        if qnn.num_inputs != n:
            raise ValueError(f"create_qnn(n) must take n={n} inputs; "
                             f"got num_inputs={qnn.num_inputs}")
        if qnn.num_weights < 1:
            raise ValueError("the quantum head has no trainable weights")
        if tuple(np.atleast_1d(qnn.output_shape)) != (1,):
            raise ValueError(f"use exactly ONE observable; got {qnn.output_shape}")

        qc = qnn.circuit
        qc_flat = qc.decompose(reps=3)
        n_qubits = qc.num_qubits
        n_gates = int(sum(qc_flat.count_ops().values()))
        depth = qc_flat.depth()
        n_weights = int(qnn.num_weights)

        train_loader, eval_loader = load_br35h()

        manual_seed(seed)
        model = FullHybridQCNN(qnn).to(device)
        optimizer = optim.NAdam(model.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        # per-epoch progress log, mirroring the paper's training_performance_*.txt
        # (epoch, mean train loss, validation accuracy) -- printed AND appended to
        # a txt in plot_dir. Validation runs every SE_FULL_EVAL_EVERY epochs (and
        # on the last one), since an eval pass over the quantum head is not free.
        os.makedirs(plot_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_iter{iteration}" if iteration is not None else ""
        log_path = os.path.join(plot_dir, f"training_qcnn_full_{stamp}{tag}.txt")
        with open(log_path, "w", encoding="utf-8") as flog:
            flog.write("epoch, train_loss, val_accuracy(%)\n")

        def _predict_eval():
            model.eval()
            p, t = [], []
            with no_grad():
                for imgs, targets in eval_loader:
                    p.extend(model(imgs.to(device)).argmax(dim=1).cpu().tolist())
                    t.extend(targets.tolist())
            return p, t

        loss_hist = []
        last_preds = last_trues = None
        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_func(model(imgs), targets)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            avg_train_loss = sum(batch_losses) / max(len(batch_losses), 1)
            loss_hist.append(avg_train_loss)

            if epoch % SE_FULL_EVAL_EVERY == 0 or epoch == epochs - 1:
                last_preds, last_trues = _predict_eval()
                val_str = f"{100.0 * accuracy_score(last_trues, last_preds):.2f}%"
            else:
                val_str = "-"
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(f"{epoch + 1}, {avg_train_loss:.4f}, {val_str}\n")
            print(f"Epoch: {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, "
                  f"Validation Accuracy: {val_str}")

        # final metrics from the last evaluation (the last epoch always evaluates;
        # guard the epochs==0 edge)
        if last_preds is None:
            last_preds, last_trues = _predict_eval()
        preds, trues = last_preds, last_trues

        acc = accuracy_score(trues, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            trues, preds, average="binary", zero_division=0)

        metrics = {
            "ok": True, "accuracy": float(acc), "f1": float(f1),
            "precision": float(prec), "recall": float(rec),
            "train_loss": round(loss_hist[-1], 4) if loss_hist else None,
            "n_qubits": n_qubits, "n_gates": n_gates, "depth": depth,
            "n_weights": n_weights,
            "n_params_total": int(sum(p.numel() for p in model.parameters())),
            "train_log": log_path,
        }
        if plot:
            metrics["plot_path"] = _save_curve(loss_hist, acc, iteration, plot_dir)

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
            metrics["error"] = ("NOT VIABLE -- head exceeds the resource budget: "
                                + "; ".join(violations))
        return metrics
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    # Train one seed head end-to-end on Br35H as a faithful reference run.
    # Point SE_DATA_DIR at your dataset; use SE_FULL_LIMIT + SE_FULL_EPOCHS small
    # for a quick smoke run (e.g. SE_FULL_LIMIT=40 SE_FULL_EPOCHS=3).
    import sys
    from stronglyentangled_task import SEED_QNNS
    name = sys.argv[1] if len(sys.argv) > 1 else "qce2_zz_realamp"
    print(f"[full] training seed {name!r} end-to-end on {SE_DATA_DIR} "
          f"(epochs={SE_FULL_EPOCHS}, batch={SE_FULL_BATCH}, lr={SE_FULL_LR}, "
          f"limit={SE_FULL_LIMIT or 'all'}, device={device})")
    m = evaluate_qcnn_full(SEED_QNNS[name], n=SE_DIM, plot=True)
    print("result:", m)
