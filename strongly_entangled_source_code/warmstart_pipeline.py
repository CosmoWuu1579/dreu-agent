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
    BATCH / SEED

RECOMMENDED (fast + avoids the unfreeze collapse entirely):
    VARIANT=Q2E2 FAST=1 WARMUP_EPOCHS=25 QUANTUM_EPOCHS=40 python warmstart_pipeline.py
"""

from __future__ import annotations

import os
import datetime

import numpy as np
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


transform = transforms.Compose([
    transforms.Resize(130), transforms.CenterCrop(128),
    transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]),
])


def make_loaders():
    train = BrainDataset(os.path.join(SE_DATA_DIR, "train"), transform)
    eval_dir = os.path.join(SE_DATA_DIR, "val")
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(SE_DATA_DIR, "test")
    ev = BrainDataset(eval_dir, transform)
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


def create_qnn(variant: str) -> EstimatorQNN:
    """Build the EstimatorQNN for a variant name like 'Q4E2'.
    QxEy -> x = #qubits (2 or 4), y = encoder (1=product Z, 2=entangled ZZ)."""
    v = variant.upper()
    n = int(v[1])                                  # Q2.. -> 2, Q4.. -> 4
    fmap = z_feature_map(n) if v.endswith("1") else zz_feature_map(n)
    if n == 2:                                      # simple RealAmplitudes VQC
        ansatz = real_amplitudes(2, reps=1)
    else:                                           # paper's conv/pool ansatz (Q4)
        ansatz = QuantumCircuit(n, name="Ansatz")
        ansatz.compose(_conv_layer(4, "c1"), list(range(4)), inplace=True)
        ansatz.compose(_pool_layer([0, 1], [2, 3], "p1"), list(range(4)), inplace=True)
        ansatz.compose(_conv_layer(2, "c2"), list(range(2, 4)), inplace=True)
        ansatz.compose(_pool_layer([0], [1], "p2"), list(range(2, 4)), inplace=True)
    qc = QuantumCircuit(n)
    qc.compose(fmap, range(n), inplace=True)
    qc.compose(ansatz, range(n), inplace=True)
    obs = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1)])   # single-qubit (local) Z
    return EstimatorQNN(circuit=qc.decompose(), observables=obs,
                        input_params=fmap.parameters, weight_params=ansatz.parameters,
                        input_gradients=True)


class QuantumHead(nn.Module):
    """fc2 -> QNN -> fc3 -> cat. fc2 auto-sizes to the QNN's input count."""
    def __init__(self, qnn):
        super().__init__()
        self.fc2 = nn.Linear(512, qnn.num_inputs)   # 512 -> 2 or 4, per variant
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)

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


def train_phase(model, train_loader, eval_loader, epochs, lr, tag, log_path):
    opt = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
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
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(f"{tag}, {epoch + 1}, {avg:.4f}, {100 * acc:.2f}%\n")
        print(f"[{tag}] epoch {epoch + 1}/{epochs}: loss={avg:.4f}  val_acc={100 * acc:.2f}%")
    return evaluate(model, eval_loader)


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


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    manual_seed(SEED); np.random.seed(SEED)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"warmstart_{VARIANT}_{stamp}.txt"
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(f"# variant={VARIANT} fast={FAST}\nphase, epoch, train_loss, val_accuracy(%)\n")
    print(f"variant={VARIANT}  device={device}  fast={FAST}  data={SE_DATA_DIR}  log={log_path}")

    train_loader, eval_loader = make_loaders()
    backbone = Backbone().to(device)

    # --- Phase 1: classical warm-up ---
    print("\n=== Phase 1: classical warm-up ===")
    cls = ClassicalNet(backbone).to(device)
    acc1, _ = train_phase(cls, train_loader, eval_loader, WARMUP_EPOCHS, LR, "warmup", log_path)
    print(f"Phase 1 done: val_acc={100 * acc1:.2f}%")

    # --- Phase 2: quantum ---
    qnn = create_qnn(VARIANT)
    print(f"\n=== Phase 2: quantum ({VARIANT}, qubits={qnn.num_inputs}) ===")
    if FAST:
        # freeze backbone, cache its features once, train the head on cached feats
        print("FAST: caching backbone features, training quantum head only")
        Ftr, ytr = cache_features(backbone, train_loader)
        Fte, yte = cache_features(backbone, eval_loader)
        head = QuantumHead(qnn).to(device)
        tl = DataLoader(TensorDataset(Ftr, ytr), batch_size=BATCH, shuffle=True)
        el = DataLoader(TensorDataset(Fte, yte), batch_size=BATCH, shuffle=False)
        acc2, f1_2 = train_phase(head, tl, el, QUANTUM_EPOCHS, LR, "q_cached", log_path)
    else:
        qnet = QuantumNet(backbone, qnn).to(device)
        if FREEZE_EPOCHS > 0:
            print(f"freezing backbone for the first {FREEZE_EPOCHS} epochs (lr={LR})")
            set_backbone_trainable(qnet, False)
            train_phase(qnet, train_loader, eval_loader, FREEZE_EPOCHS, LR, "q_frozen", log_path)
            set_backbone_trainable(qnet, True)
        # unfreeze at a MUCH lower lr -- at the full lr this destroys the warm
        # backbone in ~2 epochs (see notes.md #6)
        print(f"unfreezing backbone; fine-tuning at lr={FINETUNE_LR} (vs {LR})")
        acc2, f1_2 = train_phase(qnet, train_loader, eval_loader, QUANTUM_EPOCHS,
                                 FINETUNE_LR, "q_finetune", log_path)

    print("\n" + "=" * 60)
    print(f"FINAL {VARIANT}: accuracy={100 * acc2:.2f}%  f1={100 * f1_2:.2f}%  "
          f"(classical warm-up {100 * acc1:.2f}%)")
    print(f"log: {log_path}")
