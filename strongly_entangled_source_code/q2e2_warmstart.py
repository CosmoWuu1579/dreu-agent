#!/usr/bin/env python
"""
q2e2_warmstart.py -- fixes the Q2E2 "stuck at 50%" stall with warm-started
(staged) training.

WHY THE ORIGINAL STALLS (evidence in notes.md): the quantum head trains fine on
GOOD features (verified: 4/4 seeds perfect on separable data), but from-scratch
it sits behind a randomly-initialized CNN. That random CNN can't bootstrap good
features THROUGH the quantum bottleneck, so the joint optimization gets stuck at
a 50% local minimum whose escape depends on the seed. Fix: give the quantum head
good features to start from -- i.e. warm-start / pretrain the CNN.

HOW THIS SCRIPT DOES IT (two phases, ONE model, NO cross-file weight loading):

  Phase 1  (classical warm-up): train  BACKBONE (conv1-4 -> fc1 -> 512 feats)
           + a plain Linear(512, 2) classifier, end-to-end. This is the paper's
           CNN backbone with an ordinary head; it trains robustly and teaches the
           backbone to produce tumour-vs-healthy-discriminative features.

  Phase 2  (quantum fine-tune): KEEP the same, now-trained BACKBONE, throw away
           the Phase-1 linear head, attach the QUANTUM head (fc2: 512->2 -> QNN
           -> fc3 -> cat), and train. The quantum head now sees GOOD features from
           step 1 instead of random noise, so it escapes the 50% basin.

WHY THIS RESOLVES THE "512->2 vs 512->1" CONFUSION: we never copy weights between
mismatched layers. The shared part -- conv1-4 and fc1 (identical shapes in the
classical and quantum nets) -- is the SAME `Backbone` object reused across phases,
so it is already warm in Phase 2 with zero copying. Only the small head changes:
the Phase-1 Linear(512, 2) is discarded and the Phase-2 quantum head (fc2 512->2
-> QNN -> fc3) is trained fresh on top of the warm backbone. Standard transfer
learning: keep the backbone, swap the head.

Run:
    # dataset at <SE_DATA_DIR>/{train,val|test}/{yes,no}/  (use split_dataset.py)
    conda activate dreu && pip install torchvision      # if not already
    SE_DATA_DIR=./data/Br35H python q2e2_warmstart.py

Knobs (env):
    SE_DATA_DIR         dataset root (default ./data/Br35H)
    WARMUP_EPOCHS       Phase-1 classical epochs      (default 40)
    QUANTUM_EPOCHS      Phase-2 quantum epochs        (default 60)
    FREEZE_EPOCHS       freeze the backbone for the first N Phase-2 epochs
                        (default 10; then it unfreezes and fine-tunes)
    LR                  NAdam learning rate           (default 1e-3)
    BATCH               batch size                    (default 32)
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SE_DATA_DIR = os.environ.get("SE_DATA_DIR", "./data/Br35H")
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "40"))
QUANTUM_EPOCHS = int(os.environ.get("QUANTUM_EPOCHS", "60"))
FREEZE_EPOCHS = int(os.environ.get("FREEZE_EPOCHS", "10"))
LR = float(os.environ.get("LR", "0.001"))
BATCH = int(os.environ.get("BATCH", "32"))
SEED = int(os.environ.get("SEED", "0"))


# ---------------------------------------------------------------------------
# Dataset (the paper's, verbatim)
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
            raise FileNotFoundError(
                f"no images under {root_dir!r} (need yes/ and no/). "
                f"Set SE_DATA_DIR / run split_dataset.py.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


transform = transforms.Compose([
    transforms.Resize(130),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
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
# The SHARED backbone (conv1-4 -> fc1 -> 512 features). Identical to both the
# paper's CNN.py and CNN-Q2E2.py up to fc1 -- this is the part we warm-start.
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
        x = x.view(x.shape[0], -1)
        return F.leaky_relu(self.fc1(x))               # (N, 512) features


# ---------------------------------------------------------------------------
# The two heads that sit on top of the backbone
# ---------------------------------------------------------------------------
class ClassicalNet(nn.Module):
    """Phase 1: backbone + a plain Linear(512, 2) classifier."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def create_qnn():
    """The paper's Q2E2 quantum head: ZZ-entangled encoder + RealAmplitudes VQC.
    (Uses the default observable, exactly like CNN-Q2E2.py -- the observable is
    NOT the cause of the stall; see notes.md.)"""
    fmap = zz_feature_map(2)
    ansatz = real_amplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(fmap, inplace=True)
    qc.compose(ansatz, inplace=True)
    return EstimatorQNN(circuit=qc, input_params=fmap.parameters,
                        weight_params=ansatz.parameters, input_gradients=True)


class QuantumNet(nn.Module):
    """Phase 2: the SAME (warm) backbone + the paper's quantum head."""
    def __init__(self, backbone, qnn):
        super().__init__()
        self.backbone = backbone                        # reused, already trained
        self.fc2 = nn.Linear(512, 2)                    # -> 2 encoder inputs
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc2(self.backbone(x))
        x = self.fc3(self.qnn(x))
        return cat((x, 1 - x), -1)


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with no_grad():
        for imgs, targets in loader:
            out = model(imgs.to(device))
            preds.extend(out.argmax(dim=1).cpu().tolist())
            trues.extend(targets.tolist())
    acc = accuracy_score(trues, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        trues, preds, average="binary", zero_division=0)
    return acc, f1


def train_phase(model, train_loader, eval_loader, epochs, lr, tag, log_path):
    opt = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        losses = []
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(imgs), targets)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg = sum(losses) / max(len(losses), 1)
        acc, f1 = evaluate(model, eval_loader)
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(f"{tag}, {epoch + 1}, {avg:.4f}, {100 * acc:.2f}%\n")
        print(f"[{tag}] epoch {epoch + 1}/{epochs}: loss={avg:.4f}  val_acc={100 * acc:.2f}%")
    return evaluate(model, eval_loader)


def set_backbone_trainable(model, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad_(trainable)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    manual_seed(SEED)
    np.random.seed(SEED)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"warmstart_q2e2_{stamp}.txt"
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write("phase, epoch, train_loss, val_accuracy(%)\n")
    print(f"device={device}  data={SE_DATA_DIR}  log={log_path}")

    train_loader, eval_loader = make_loaders()
    backbone = Backbone().to(device)

    # --- Phase 1: classical warm-up (learn good features) ---
    print("\n=== Phase 1: classical warm-up (backbone + Linear head) ===")
    cls = ClassicalNet(backbone).to(device)
    acc1, f1_1 = train_phase(cls, train_loader, eval_loader,
                             WARMUP_EPOCHS, LR, "warmup", log_path)
    print(f"Phase 1 done: val_acc={100 * acc1:.2f}%  (backbone now warm)")

    # --- Phase 2: attach quantum head to the SAME warm backbone ---
    print("\n=== Phase 2: quantum fine-tune (warm backbone + quantum head) ===")
    qnet = QuantumNet(backbone, create_qnn()).to(device)   # backbone reused, warm
    if FREEZE_EPOCHS > 0:                                   # let the head catch up first
        print(f"freezing backbone for the first {FREEZE_EPOCHS} epochs")
        set_backbone_trainable(qnet, False)
        train_phase(qnet, train_loader, eval_loader,
                    FREEZE_EPOCHS, LR, "q_frozen", log_path)
        set_backbone_trainable(qnet, True)                 # unfreeze, fine-tune together
    acc2, f1_2 = train_phase(qnet, train_loader, eval_loader,
                             QUANTUM_EPOCHS, LR, "q_finetune", log_path)

    print("\n" + "=" * 60)
    print(f"FINAL (quantum hybrid): accuracy={100 * acc2:.2f}%  f1={100 * f1_2:.2f}%")
    print(f"(classical warm-up reached {100 * acc1:.2f}% for reference)")
    print(f"per-epoch log: {log_path}")
