#!/usr/bin/env python
"""
warmstart_pipeline.py -- warm-started (staged) training for the strongly-entangled
variants: Q2E1, Q2E2, Q4E1, Q4E2 (Q8* too, but slow).

WHY (full evidence in notes.md): the paper's hybrid stalls at exactly 50% because
a randomly-initialized CNN cannot bootstrap useful features THROUGH the 1-number
quantum bottleneck, so the joint optimization sits in a 50% local minimum whose
escape is a seed lottery. The quantum head itself is fine -- on good features it
trains robustly (measured: 4/4 seeds perfect). Fix: give it good features first,
i.e. warm-start the CNN.

HOW -- two phases, ONE model, NO cross-file weight loading:
    Phase 1  classical warm-up: Backbone (conv1-4 -> fc1 -> 512 feats) + a plain
             Linear(512, 2) head, trained end-to-end. Learns good tumour-vs-
             healthy features (this part trains robustly).
    Phase 2  quantum: KEEP the same, now-warm Backbone, drop the Phase-1 linear
             head, attach the variant's quantum head (fc2 -> QNN -> fc3 -> cat),
             and train. The head now sees GOOD features instead of random noise.

The shared part (conv1-4 + fc1) is the SAME `Backbone` object reused across both
phases, so it is already warm in Phase 2 with zero weight copying -- which also
sidesteps the fc2 shape difference (512->1 in the paper's classical CNN.py vs
512->2/4 in the quantum nets): the Phase-1 head is simply discarded and the
quantum fc2 is trained fresh. Standard transfer learning: keep the backbone,
swap the head.

ONE FILE, ALL VARIANTS: the only thing that differs between variants is the
quantum head, which is parameterized:

    VARIANT=Q4E2 SE_DATA_DIR=./data/Br35H python warmstart_pipeline.py

    * encoder E1 vs E2  -> z_feature_map (product) vs zz_feature_map (entangled)
    * width Q2 vs Q4    -> 2 vs 4 qubits; Q2 uses a RealAmplitudes VQC, Q4 uses
                           the paper's conv/pool ansatz
    * fc2 width         -> auto-sized to qnn.num_inputs (2 or 4). No manual edits.

SPEED (FAST=1): after Phase 1, freeze the warm backbone and CACHE its 512-d
features once, then train the quantum head on the cached features -- no CNN
forward/backward in Phase 2, which is the big time saver. Trade-off: the backbone
is not fine-tuned in Phase 2 (it's already good from Phase 1), so accuracy may be
marginally lower for a large speedup. Note the quantum head runs on a CPU
statevector with parameter-shift gradients, so IT -- not the GPU -- is the
bottleneck: cut time by lowering QUANTUM_EPOCHS and using fewer training images.

Knobs (env):
    VARIANT        Q2E1 | Q2E2 | Q4E1 | Q4E2   (default Q2E2)
    SE_DATA_DIR    dataset root (default ./data/Br35H)
    WARMUP_EPOCHS  Phase-1 classical epochs     (default 40)
    QUANTUM_EPOCHS Phase-2 quantum epochs       (default 40)
    FREEZE_EPOCHS  freeze backbone for first N Phase-2 epochs (default 10;
                   ignored in FAST mode, where the backbone is always frozen)
    FAST           1 -> cache features, train head only (fast)   (default 0)
    LR             training lr for Phase 1 + the frozen head     (default 1e-3)
    FINETUNE_LR    lr used AFTER unfreezing the backbone         (default LR/100)
                   -- at the full LR the unfreeze causes catastrophic forgetting
                      (89% -> 51% in 2 epochs; notes.md #6). Not used in FAST.
    AUGMENT        1 -> mild train-time augmentation (flip / +-10deg / shift+scale)
                   (default 1). Overfitting is the binding constraint here
                   (train loss -> 0.04 while val plateaus ~95%), and augmentation
                   is the paper's own recommended fix. TRAIN split only; eval is
                   always the clean transform. No real effect under FAST (features
                   are cached once, so the augmentation cannot vary per epoch).
    EARLY_STOP     1 -> stop a phase after PATIENCE epochs with no val improvement
                   (default 1; PATIENCE default 8). Set EARLY_STOP=0 for old behaviour.
    SCHEDULE       1 -> ReduceLROnPlateau on val acc (default 1), tuned by
                   SCHEDULE_FACTOR (0.5) / SCHEDULE_PATIENCE (3). SCHEDULE=0 disables.
                   Both toggles are independent.
    BATCH / SEED

Each run writes runs/<VARIANT>_<stamp>/ containing training.txt (per-epoch log),
config.txt (settings + results), training.png (loss + accuracy, phases and the
best epoch marked) and best_quantum.pt (weights at the best quantum epoch).
The BEST val accuracy is what to report -- the last epoch usually is not the best.

RECOMMENDED (the frozen phase does the real work; fine-tuning adds ~nothing):
    VARIANT=Q2E2 WARMUP_EPOCHS=25 FREEZE_EPOCHS=40 QUANTUM_EPOCHS=5 python warmstart_pipeline.py
"""

from __future__ import annotations

import os
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless (cluster-safe): save figures, never open a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cat, no_grad, manual_seed
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from torchvision import transforms

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map, zz_feature_map, real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANT = os.environ.get("VARIANT", "Q2E2").upper()
SE_DATA_DIR = os.environ.get("SE_DATA_DIR", "./data/Br35H")
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "40"))
QUANTUM_EPOCHS = int(os.environ.get("QUANTUM_EPOCHS", "40"))
FREEZE_EPOCHS = int(os.environ.get("FREEZE_EPOCHS", "10"))
FAST = os.environ.get("FAST", "0") != "0"
LR = float(os.environ.get("LR", "0.001"))
# Fine-tuning a CONVERGED backbone must use a much smaller lr than training it
# from scratch: at the same lr, unfreezing wrecks the learned features in ~2
# epochs (catastrophic forgetting -- measured, see notes.md evidence #6).
FINETUNE_LR = float(os.environ.get("FINETUNE_LR", str(LR / 100)))
BATCH = int(os.environ.get("BATCH", "32"))
SEED = int(os.environ.get("SEED", "0"))
# Mild geometric augmentation on the TRAIN split only. Overfitting is the binding
# constraint on this task (train loss -> ~0.04 while val plateaus ~95%), and
# augmentation is the paper's own recommended fix for it.
AUGMENT = os.environ.get("AUGMENT", "1") != "0"

# --- both of these are independently toggleable; set to 0 for the old behaviour ---
# EARLY_STOP: halt a phase after PATIENCE epochs with no val improvement. The
# frozen quantum phase peaks early (~ep 9) then DEGRADES for 30 more epochs --
# wasting the most expensive epochs in the run.
EARLY_STOP = os.environ.get("EARLY_STOP", "1") != "0"
PATIENCE = int(os.environ.get("PATIENCE", "8"))
# SCHEDULE: ReduceLROnPlateau -- halve the lr when val accuracy stalls. The tiny
# quantum head (4 weights) oscillates at LR=1e-3 (val swinging 86<->97 while the
# loss barely moves); decaying the lr lets it settle at its peak.
SCHEDULE = os.environ.get("SCHEDULE", "1") != "0"
SCHEDULE_FACTOR = float(os.environ.get("SCHEDULE_FACTOR", "0.5"))
SCHEDULE_PATIENCE = int(os.environ.get("SCHEDULE_PATIENCE", "3"))

# --- quantum-head capacity knobs. BOTH DEFAULT TO THE PAPER'S EXACT BEHAVIOUR ---
# OBSERVABLES: what the head reads off the state. The paper reads ONE scalar,
# which is the measured bottleneck (quantum train loss floors ~10x above the
# classical head's). More observables cost nothing extra to simulate.
#   single      -> [Z on last qubit]              1 output   (paper; DEFAULT)
#   local       -> [Z on every qubit]             n outputs
#   local_corr  -> local + [ZZ adjacent pairs]    n + (n-1) outputs
OBSERVABLES = os.environ.get("OBSERVABLES", "single").strip().lower()
# REUPLOAD: data re-uploading depth. 1 = encode once then one ansatz (paper;
# DEFAULT). k>1 interleaves encode->ansatz k times, re-injecting x at every layer
# -- the standard way to make a small VQC much more expressive.
REUPLOAD = int(os.environ.get("REUPLOAD", "1"))


# ---------------------------------------------------------------------------
# Dataset (the paper's)
# ---------------------------------------------------------------------------
class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images, self.labels = [], []
        for sub, label in (("yes", 1), ("no", 0)):
            d = os.path.join(root_dir, sub)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(d, f))
                    self.labels.append(label)
        if not self.images:
            raise FileNotFoundError(f"no images under {root_dir!r} (need yes/ and no/).")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transform(augment: bool):
    """Eval transform = the paper's exactly. The train transform optionally adds
    mild geometric augmentation (flip / +-10 deg / small shift+scale), applied to
    the PIL image before ToTensor. Kept conservative so the anatomy stays
    plausible."""
    steps = [transforms.Resize(130), transforms.CenterCrop(128)]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                    scale=(0.95, 1.05)),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    return transforms.Compose(steps)


def make_loaders(augment: bool = AUGMENT):
    """(train_loader, eval_loader). Augmentation applies to TRAIN ONLY -- eval is
    always the clean paper transform, so scores stay comparable across runs."""
    train = BrainDataset(os.path.join(SE_DATA_DIR, "train"), build_transform(augment))
    eval_dir = os.path.join(SE_DATA_DIR, "val")
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(SE_DATA_DIR, "test")
    ev = BrainDataset(eval_dir, build_transform(False))
    return (DataLoader(train, batch_size=BATCH, shuffle=True),
            DataLoader(ev, batch_size=BATCH, shuffle=False))


# ---------------------------------------------------------------------------
# Shared backbone (conv1-4 -> fc1 -> 512 features)
# ---------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = self.dropout(x)
        return F.leaky_relu(self.fc1(x.view(x.shape[0], -1)))


class ClassicalNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# The paper's conv/pool ansatz (needed for the 4- and 8-qubit variants)
# ---------------------------------------------------------------------------
def _conv_circuit(p):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1); qc.cx(1, 0)
    qc.rz(p[0], 0); qc.ry(p[1], 1); qc.cx(0, 1); qc.ry(p[2], 1)
    qc.cx(1, 0); qc.rz(np.pi / 2, 0)
    return qc


def _conv_layer(n, prefix):
    qc = QuantumCircuit(n); qubits = list(range(n)); i = 0
    p = ParameterVector(prefix, length=n * 3)
    for a, b in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(_conv_circuit(p[i:i + 3]), [a, b]); i += 3
    for a, b in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(_conv_circuit(p[i:i + 3]), [a, b]); i += 3
    return qc


def _pool_circuit(p):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1); qc.cx(1, 0)
    qc.rz(p[0], 0); qc.ry(p[1], 1); qc.cx(0, 1); qc.ry(p[2], 1)
    return qc


def _pool_layer(sources, sinks, prefix):
    n = len(sources) + len(sinks); qc = QuantumCircuit(n); i = 0
    p = ParameterVector(prefix, length=n // 2 * 3)
    for s, t in zip(sources, sinks):
        qc = qc.compose(_pool_circuit(p[i:i + 3]), [s, t]); i += 3
    return qc


def _pauli(n: int, spec: dict) -> str:
    """Pauli string for {qubit: 'Z'}. Qiskit puts the HIGHEST qubit leftmost, so
    n=2, {0:'Z'} -> 'IZ' (Z on qubit 0); {1:'Z'} -> 'ZI'."""
    s = ["I"] * n
    for q, p in spec.items():
        s[n - 1 - q] = p
    return "".join(s)


def build_observables(n: int, mode: str):
    """What the head reads off the state (see the OBSERVABLES knob).
    mode='single' reproduces the paper exactly: Z on the last qubit."""
    mode = (mode or "single").lower()
    if mode == "single":
        return [SparsePauliOp.from_list([(_pauli(n, {n - 1: "Z"}), 1)])]
    if mode not in ("local", "local_corr"):
        raise ValueError(f"OBSERVABLES must be single|local|local_corr, got {mode!r}")
    obs = [SparsePauliOp.from_list([(_pauli(n, {q: "Z"}), 1)]) for q in range(n)]
    if mode == "local_corr":
        obs += [SparsePauliOp.from_list([(_pauli(n, {q: "Z", q + 1: "Z"}), 1)])
                for q in range(n - 1)]
    return obs


def build_ansatz(n: int, prefix: str) -> QuantumCircuit:
    """The trainable block. `prefix` keeps weights unique per re-upload layer."""
    if n == 2:                                      # simple RealAmplitudes VQC
        return real_amplitudes(2, reps=1, parameter_prefix=prefix)
    a = QuantumCircuit(n, name="Ansatz")            # paper's conv/pool ansatz (Q4)
    a.compose(_conv_layer(4, f"{prefix}c1"), list(range(4)), inplace=True)
    a.compose(_pool_layer([0, 1], [2, 3], f"{prefix}p1"), list(range(4)), inplace=True)
    a.compose(_conv_layer(2, f"{prefix}c2"), list(range(2, 4)), inplace=True)
    a.compose(_pool_layer([0], [1], f"{prefix}p2"), list(range(2, 4)), inplace=True)
    return a


def create_qnn(variant: str, obs_mode: str = None, reupload: int = None) -> EstimatorQNN:
    """Build the EstimatorQNN for a variant name like 'Q4E2'.
    QxEy -> x = #qubits (2 or 4), y = encoder (1=product Z, 2=entangled ZZ).

    With the defaults (OBSERVABLES='single', REUPLOAD=1) this builds EXACTLY the
    paper's circuit: encode once -> one ansatz -> read one Z. The knobs only add
    structure on top.

    reupload > 1 = DATA RE-UPLOADING: the SAME feature-map object is composed
    again between trainable blocks (encode -> ansatz -> encode -> ansatz ...), so
    the same input Parameters are re-injected at every layer (that is what makes
    it re-uploading rather than new inputs); each layer gets its own weights.
    """
    obs_mode = OBSERVABLES if obs_mode is None else obs_mode
    reupload = REUPLOAD if reupload is None else reupload
    v = variant.upper()
    n = int(v[1])                                   # Q2.. -> 2, Q4.. -> 4
    fmap = z_feature_map(n) if v.endswith("1") else zz_feature_map(n)  # built ONCE

    qc = QuantumCircuit(n)
    weight_params = []
    for layer in range(max(1, reupload)):
        qc.compose(fmap, range(n), inplace=True)    # same Parameters -> re-upload
        ansatz = build_ansatz(n, prefix=f"w{layer}_")
        qc.compose(ansatz, range(n), inplace=True)
        weight_params += list(ansatz.parameters)

    return EstimatorQNN(circuit=qc.decompose(),
                        observables=build_observables(n, obs_mode),
                        input_params=list(fmap.parameters),
                        weight_params=weight_params,
                        input_gradients=True)


class QuantumHead(nn.Module):
    """fc2 -> QNN -> fc3 -> cat. fc2 auto-sizes to the QNN's input count."""
    def __init__(self, qnn):
        super().__init__()
        self.fc2 = nn.Linear(512, qnn.num_inputs)   # 512 -> 2 or 4, per variant
        self.qnn = TorchConnector(qnn)
        # sized to the number of observables: 1 with the paper's default, more
        # when OBSERVABLES=local/local_corr widens the readout
        n_out = int(np.atleast_1d(qnn.output_shape)[0])
        self.fc3 = nn.Linear(n_out, 1)

    def forward(self, feats):                        # feats = 512-d backbone output
        x = self.fc3(self.qnn(self.fc2(feats)))
        return cat((x, 1 - x), -1)


class QuantumNet(nn.Module):
    """Backbone + QuantumHead (end-to-end path, used when FAST is off)."""
    def __init__(self, backbone, qnn):
        super().__init__()
        self.backbone = backbone
        self.head = QuantumHead(qnn)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with no_grad():
        for xb, yb in loader:
            preds.extend(model(xb.to(device)).argmax(dim=1).cpu().tolist())
            trues.extend(yb.tolist())
    acc = accuracy_score(trues, preds)
    _, _, f1, _ = precision_recall_fscore_support(trues, preds, average="binary",
                                                  zero_division=0)
    return acc, f1


def train_phase(model, train_loader, eval_loader, epochs, lr, tag, log_path,
                history=None, best=None, ckpt_path=None):
    """Train for `epochs`, logging per epoch.

    history   : list, collects {tag, epoch, loss, acc} across ALL phases (plots).
    best      : mutable dict {acc, tag, epoch} tracking the best val accuracy seen
                so far ACROSS the phases it is passed to. The last epoch is often
                NOT the best (val accuracy wanders while train loss keeps falling
                -- overfitting), so this is the number worth reporting.
    ckpt_path : if given, the model's state_dict is saved whenever `best` improves.

    Honors the EARLY_STOP / SCHEDULE toggles (both independently switchable).
    """
    opt = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    sched = (optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=SCHEDULE_FACTOR, patience=SCHEDULE_PATIENCE)
        if SCHEDULE else None)
    phase_best, stale = -1.0, 0          # phase-local, drives early stopping
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg = sum(losses) / max(len(losses), 1)
        acc, _ = evaluate(model, eval_loader)
        if history is not None:
            history.append({"tag": tag, "epoch": epoch + 1, "loss": avg, "acc": acc})

        marker = ""
        if best is not None and acc > best["acc"]:      # global (cross-phase) best
            best.update(acc=acc, tag=tag, epoch=epoch + 1)
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
            marker = "  <- best"

        if sched is not None:                           # decay lr when val stalls
            prev_lr = opt.param_groups[0]["lr"]
            sched.step(acc)
            new_lr = opt.param_groups[0]["lr"]
            if new_lr < prev_lr:
                marker += f"  (lr {prev_lr:.2g} -> {new_lr:.2g})"

        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(f"{tag}, {epoch + 1}, {avg:.4f}, {100 * acc:.2f}%\n")
        print(f"[{tag}] epoch {epoch + 1}/{epochs}: loss={avg:.4f}  "
              f"val_acc={100 * acc:.2f}%{marker}")

        if acc > phase_best + 1e-9:
            phase_best, stale = acc, 0
        else:
            stale += 1
        if EARLY_STOP and PATIENCE > 0 and stale >= PATIENCE:
            print(f"[{tag}] early stop: no val improvement in {PATIENCE} epochs "
                  f"(phase best {100 * phase_best:.2f}%)")
            break
    return evaluate(model, eval_loader)


def save_plots(history, run_dir, acc1=None, acc2=None, best_q=None):
    """Train loss + validation accuracy vs cumulative epoch, with the phase
    boundaries marked -- the transitions (warm-up -> frozen -> fine-tune) are
    where the interesting behaviour lives. `best_q` (if given) is starred on the
    accuracy panel."""
    if not history:
        return None
    xs = list(range(1, len(history) + 1))
    losses = [h["loss"] for h in history]
    accs = [100 * h["acc"] for h in history]
    tags = [h["tag"] for h in history]

    # contiguous spans of the same phase -> boundaries + label positions
    spans, start = [], 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] != tags[start]:
            spans.append((tags[start], start + 1, i))     # (tag, first, last)
            start = i
    bounds = [s[1] - 0.5 for s in spans[1:]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(xs, losses, color="#2563eb", lw=1.6)
    ax1.set_xlabel("epoch (cumulative)"); ax1.set_ylabel("train loss")
    ax1.set_title("Training loss")

    ax2.plot(xs, accs, color="#d97706", lw=1.6)
    ax2.axhline(50, ls=":", c="0.6", lw=1)               # chance line for binary
    ax2.text(xs[0], 51, "chance (50%)", fontsize=7, color="0.45")
    ax2.set_xlabel("epoch (cumulative)"); ax2.set_ylabel("val accuracy (%)")
    ax2.set_ylim(0, 100); ax2.set_title("Validation accuracy")

    # star the best quantum epoch (the reported number -- the LAST epoch usually
    # is not the best once overfitting sets in)
    if best_q and best_q.get("acc", -1) >= 0:
        for i, h in enumerate(history):
            if h["tag"] == best_q["tag"] and h["epoch"] == best_q["epoch"]:
                bx, by = i + 1, 100 * h["acc"]
                ax2.plot([bx], [by], marker="*", ms=15, color="#16a34a", zorder=5)
                ax2.annotate(f"best {by:.1f}%", (bx, by), textcoords="offset points",
                             xytext=(0, -17), ha="center", fontsize=8, color="#16a34a")
                break

    for ax in (ax1, ax2):
        for b in bounds:
            ax.axvline(b, ls="--", c="0.5", lw=1)
        ax.grid(alpha=0.3)
        top = ax.get_ylim()[1]
        for tag, a, b in spans:                           # phase name per span
            ax.text((a + b) / 2, top * 0.97, tag, fontsize=7, color="0.35",
                    ha="center", va="top")

    title = f"{VARIANT}"
    if acc1 is not None and acc2 is not None:
        title += (f"  |  classical warm-up {100 * acc1:.1f}%  ->  "
                  f"quantum {100 * acc2:.1f}%")
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(run_dir, "training.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def cache_features(backbone, loader):
    """Run the (frozen) backbone once and return (features, labels) tensors."""
    backbone.eval()
    feats, labels = [], []
    with no_grad():
        for xb, yb in loader:
            feats.append(backbone(xb.to(device)).cpu())
            labels.append(yb)
    return torch.cat(feats), torch.cat(labels)


def set_backbone_trainable(net, flag):
    for p in net.backbone.parameters():
        p.requires_grad_(flag)


def write_config(path, train_loader, eval_loader, qnn):
    """Snapshot every setting that affects this run, next to its training log, so
    any result can be reproduced from the folder it sits in."""
    import qiskit
    import qiskit_machine_learning as qml_pkg
    n = qnn.num_inputs
    lines = [
        f"# warmstart_pipeline run config -- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "[variant]",
        f"VARIANT                 = {VARIANT}",
        f"qubits (qnn.num_inputs) = {n}",
        f"quantum weights (#Para) = {qnn.num_weights}",
        f"OBSERVABLES             = {OBSERVABLES}  "
        f"({int(np.atleast_1d(qnn.output_shape)[0])} readout(s))",
        f"REUPLOAD                = {REUPLOAD}"
        + ("  (paper default: encode once)" if REUPLOAD <= 1 else
           f"  (encoder re-applied {REUPLOAD}x)"),
        f"encoder                 = " + ("z_feature_map (E1, product, no entanglement)"
                                         if VARIANT.endswith("1") else
                                         "zz_feature_map (E2, pairwise-entangled)"),
        f"ansatz                  = " + ("real_amplitudes(reps=1)" if n == 2
                                         else "conv/pool (paper's QCNN ansatz)"),
        "",
        "[data]",
        f"SE_DATA_DIR             = {os.path.abspath(SE_DATA_DIR)}",
        f"train images            = {len(train_loader.dataset)}",
        f"eval images             = {len(eval_loader.dataset)}",
        f"BATCH                   = {BATCH}",
        f"AUGMENT (train only)    = {AUGMENT}"
        + ("  (no effect under FAST: features cached once)" if (AUGMENT and FAST) else ""),
        f"EARLY_STOP / PATIENCE   = {EARLY_STOP} / {PATIENCE}",
        f"SCHEDULE                = {SCHEDULE}"
        + (f"  (ReduceLROnPlateau factor={SCHEDULE_FACTOR} patience={SCHEDULE_PATIENCE})"
           if SCHEDULE else ""),
        "",
        "[training]",
        f"WARMUP_EPOCHS           = {WARMUP_EPOCHS}",
        f"FREEZE_EPOCHS           = {FREEZE_EPOCHS}",
        f"QUANTUM_EPOCHS          = {QUANTUM_EPOCHS}",
        f"FAST                    = {FAST}",
        f"LR                      = {LR}",
        f"FINETUNE_LR             = {FINETUNE_LR}",
        f"SEED                    = {SEED}",
        "",
        "[env]",
        f"device                  = {device}",
        f"torch                   = {torch.__version__}",
        f"qiskit                  = {qiskit.__version__}",
        f"qiskit-machine-learning = {qml_pkg.__version__}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    manual_seed(SEED); np.random.seed(SEED)

    # one folder per run: runs/<VARIANT>_<stamp>/ holds the training log + the
    # config snapshot that produced it
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{VARIANT}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "training.txt")
    config_path = os.path.join(run_dir, "config.txt")

    train_loader, eval_loader = make_loaders()
    qnn = create_qnn(VARIANT)                    # built early so config can log it
    write_config(config_path, train_loader, eval_loader, qnn)
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write("phase, epoch, train_loss, val_accuracy(%)\n")
    print(f"run dir: {os.path.abspath(run_dir)}")
    print(f"variant={VARIANT}  device={device}  fast={FAST}  data={SE_DATA_DIR}")

    backbone = Backbone().to(device)
    history = []                                 # every epoch, every phase -> plots
    best_cls = {"acc": -1.0, "tag": None, "epoch": None}   # best classical epoch
    best_q = {"acc": -1.0, "tag": None, "epoch": None}     # best quantum epoch

    # --- Phase 1: classical warm-up ---
    print("\n=== Phase 1: classical warm-up ===")
    cls = ClassicalNet(backbone).to(device)
    acc1, _ = train_phase(cls, train_loader, eval_loader, WARMUP_EPOCHS, LR, "warmup",
                          log_path, history, best=best_cls)
    print(f"Phase 1 done: final={100 * acc1:.2f}%  best={100 * best_cls['acc']:.2f}%")

    # --- Phase 2: quantum ---
    print(f"\n=== Phase 2: quantum ({VARIANT}, qubits={qnn.num_inputs}) ===")
    ckpt = os.path.join(run_dir, "best_quantum.pt")
    if FAST:
        # freeze backbone, cache its features once, train the head on cached feats
        print("FAST: caching backbone features, training quantum head only")
        if AUGMENT:
            print("  NOTE: features are cached ONCE, so augmentation cannot vary per "
                  "epoch -- caching from the CLEAN transform. Use FAST=0 to actually "
                  "benefit from AUGMENT.")
        clean_train, _ = make_loaders(augment=False)
        Ftr, ytr = cache_features(backbone, clean_train)
        Fte, yte = cache_features(backbone, eval_loader)
        head = QuantumHead(qnn).to(device)
        tl = DataLoader(TensorDataset(Ftr, ytr), batch_size=BATCH, shuffle=True)
        el = DataLoader(TensorDataset(Fte, yte), batch_size=BATCH, shuffle=False)
        acc2, f1_2 = train_phase(head, tl, el, QUANTUM_EPOCHS, LR, "q_cached",
                                 log_path, history, best=best_q, ckpt_path=ckpt)
    else:
        qnet = QuantumNet(backbone, qnn).to(device)
        if FREEZE_EPOCHS > 0:
            print(f"freezing backbone for the first {FREEZE_EPOCHS} epochs (lr={LR})")
            set_backbone_trainable(qnet, False)
            train_phase(qnet, train_loader, eval_loader, FREEZE_EPOCHS, LR, "q_frozen",
                        log_path, history, best=best_q, ckpt_path=ckpt)
            set_backbone_trainable(qnet, True)
        # unfreeze at a MUCH lower lr -- at the full lr this destroys the warm
        # backbone in ~2 epochs (see notes.md #6)
        print(f"unfreezing backbone; fine-tuning at lr={FINETUNE_LR} (vs {LR})")
        acc2, f1_2 = train_phase(qnet, train_loader, eval_loader, QUANTUM_EPOCHS,
                                 FINETUNE_LR, "q_finetune", log_path, history,
                                 best=best_q, ckpt_path=ckpt)

    # final results land in BOTH the training log (as trailing comments) and the
    # config, so each run folder is self-describing at a glance
    summary = (f"# FINAL {VARIANT}: quantum BEST={100 * best_q['acc']:.2f}% "
               f"(@{best_q['tag']} ep{best_q['epoch']})  last={100 * acc2:.2f}%  "
               f"f1={100 * f1_2:.2f}%  |  classical best={100 * best_cls['acc']:.2f}%")
    with open(log_path, "a", encoding="utf-8") as flog:
        flog.write(summary + "\n")
    with open(config_path, "a", encoding="utf-8") as f:
        f.write(f"\n[results]\n"
                f"classical_warmup_best   = {100 * best_cls['acc']:.2f}%  "
                f"(ep{best_cls['epoch']})\n"
                f"classical_warmup_last   = {100 * acc1:.2f}%\n"
                f"quantum_best            = {100 * best_q['acc']:.2f}%  "
                f"({best_q['tag']} ep{best_q['epoch']})   <- report this\n"
                f"quantum_last            = {100 * acc2:.2f}%\n"
                f"quantum_f1_last         = {100 * f1_2:.2f}%\n"
                f"best_checkpoint         = best_quantum.pt\n")

    plot_path = save_plots(history, run_dir, acc1, acc2, best_q)

    print("\n" + "=" * 60)
    print(summary.lstrip("# "))
    print(f"run dir: {os.path.abspath(run_dir)}")
    print(f"  training.txt      (per-epoch log)")
    print(f"  config.txt        (settings + results)")
    print(f"  best_quantum.pt   (weights at the best quantum epoch)")
    if plot_path:
        print(f"  training.png      (loss + val accuracy, phases + best marked)")
