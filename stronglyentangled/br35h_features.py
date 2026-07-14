"""
Real Br35H features for the FROZEN evaluator.

This is "the swap": instead of the qiskit `ad_hoc` stand-in vectors, run every MRI
through a FROZEN 4-conv CNN backbone (the paper's, up to the 512-d fc1 layer) and
return its feature vectors + labels -- a drop-in replacement for
stronglyentangled_eval.load_se_data. That turns the frozen task from a generic VQC
toy into the paper's actual question -- "does an entangled TRAINED head separate
the CNN's tumor/healthy features better than a product one?" -- WITHOUT paying to
train the CNN too (that is the full evaluator, SE_FULL=1).

Enable it with SE_FEATURES=br35h (see stronglyentangled_task). Nothing downstream
changes: the head's fc2 auto-sizes to the feature width (512), `n` stays the qubit
count, and the QNN / training loop / logging are identical -- only `data` differs.

The CNN is FROZEN. By default it is RANDOMLY initialized: a fixed random feature
extractor (think random projection / random kitchen sinks) that keeps the problem
non-trivial so entanglement can matter. Point SE_CNN_WEIGHTS at a trained cnn.pt
to use LEARNED features instead -- but beware the "ceiling" trap: a well-trained
CNN can make the features so separable that every head ties; an under-trained
checkpoint is safer.

Env knobs:
    SE_DATA_DIR      Br35H root: <dir>/train/{yes,no}, and val/ or test/{yes,no}
    SE_CNN_WEIGHTS   path to a CNN state_dict (.pt); default = random frozen CNN
    SE_FEATURE_SEED  seed for the random CNN init (default 0)
    SE_FEATURE_LIMIT cap images PER SPLIT for a quick run (0 = all)
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import no_grad

# NOTE: torchvision / PIL are imported LAZILY (inside the loader) so this module
# imports fine without them -- you only need them when actually reading images.

SE_DATA_DIR = os.environ.get("SE_DATA_DIR", "./data/Br35H")
SE_CNN_WEIGHTS = os.environ.get("SE_CNN_WEIGHTS", "").strip() or None
SE_FEATURE_SEED = int(os.environ.get("SE_FEATURE_SEED", "0"))
SE_FEATURE_LIMIT = int(os.environ.get("SE_FEATURE_LIMIT", "0"))
SE_IMG = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# The paper's 4-conv feature extractor, up to the 512-d fc1 output. This is
# exactly FullHybridQCNN's backbone (conv1..conv4 -> fc1), minus fc2/qnn/fc3 --
# i.e. the classical part that the frozen proxy assumes is already trained.
# ---------------------------------------------------------------------------
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)     # 128x128 -> 8x8 after 4 pools

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        return F.leaky_relu(self.fc1(x))           # (N, 512) features


def _get_transform():
    from torchvision import transforms          # lazy: only needed with real data
    return transforms.Compose([
        transforms.Resize(130),
        transforms.CenterCrop(SE_IMG),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def _load_split_images(split_dir: str, limit: int = 0):
    from PIL import Image                        # lazy
    tf = _get_transform()
    imgs, labels = [], []
    for sub, label in (("yes", 1), ("no", 0)):
        d = os.path.join(split_dir, sub)
        if not os.path.isdir(d):
            continue
        names = [f for f in os.listdir(d)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if limit:
            names = names[:limit]
        for f in names:
            imgs.append(tf(Image.open(os.path.join(d, f)).convert("L")))
            labels.append(label)
    if not imgs:
        raise FileNotFoundError(
            f"no images under {split_dir!r} (expected yes/ and no/ subfolders). "
            f"Set SE_DATA_DIR to your Br35H split (see split_dataset.py).")
    return torch.stack(imgs), np.asarray(labels, dtype=int)


_CACHE: dict = {}


def load_br35h_features(data_dir: str = SE_DATA_DIR,
                        weights: str | None = SE_CNN_WEIGHTS,
                        seed: int = SE_FEATURE_SEED,
                        limit: int = SE_FEATURE_LIMIT):
    """(Xtr, ytr, Xte, yte) where each X row is a 512-d FROZEN-CNN feature vector,
    standardized with train-set statistics. Computed ONCE and cached (keyed by
    dir/weights/seed/limit), so repeated evaluations reuse it."""
    key = (os.path.abspath(data_dir), weights, seed, limit)
    if key in _CACHE:
        return _CACHE[key]

    torch.manual_seed(seed)
    cnn = CNNBackbone().to(device).eval()
    if weights:                                    # else: random-frozen extractor
        cnn.load_state_dict(torch.load(weights, map_location=device), strict=False)
    for p in cnn.parameters():
        p.requires_grad_(False)

    def _features(split: str):
        imgs, y = _load_split_images(os.path.join(data_dir, split), limit)
        chunks = []
        with no_grad():
            for i in range(0, len(imgs), 64):      # batch to bound memory
                chunks.append(cnn(imgs[i:i + 64].to(device)).cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32), y

    Xtr, ytr = _features("train")
    eval_split = "val" if os.path.isdir(os.path.join(data_dir, "val")) else "test"
    Xte, yte = _features(eval_split)

    # standardize with TRAIN stats -- a fixed classical rescale (fc2 still learns);
    # keeps the encoder's input angles in a sane range across the 512 dims
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    out = (Xtr, ytr, Xte, yte)
    _CACHE[key] = out
    return out


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_br35h_features()
    print(f"train features {Xtr.shape}  labels {ytr.shape} "
          f"({int(ytr.sum())} tumor / {len(ytr) - int(ytr.sum())} healthy)")
    print(f"eval  features {Xte.shape}  labels {yte.shape}")
    print(f"feature width = {Xtr.shape[1]}  (fc2 maps this -> n qubits)")
    print(f"CNN weights   = {'RANDOM-frozen' if not SE_CNN_WEIGHTS else SE_CNN_WEIGHTS}")
