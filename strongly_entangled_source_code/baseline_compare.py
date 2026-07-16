#!/usr/bin/env python
"""
baseline_compare.py -- compare the ORIGINAL (paper) training against our
WARM-START training on the SAME data / circuit / eval, across MANY seeds and any
variant (Q2E1 / Q2E2 / Q4E1 / Q4E2).

ORIGINAL == the paper's CNN-*.py exactly: build the FULL hybrid and train it
end-to-end FROM SCRATCH, one optimizer (NAdam, ORIG_LR=1e-5), one phase. NO
warm-up, NO freezing, NO augmentation (the paper used none), NO lr decay, NO
early stopping.

WARM-START == our method, all three phases: classical warm-up -> frozen quantum
head -> unfreeze + gentle fine-tune.

Both reuse warmstart_pipeline's building blocks (Backbone, quantum head, data,
eval) and the SAME clean fixed-epoch training loop (`_train`) -- deliberately WITH
lr decay / early stopping OFF, so every seed runs the full length and the curves
are directly comparable and averageable. (Those instrumented features live in
warmstart_pipeline.py; here we want clean, complete curves.)

The circuit is PINNED to the paper's (single observable, reupload=1) for BOTH
methods regardless of env, so "original" always means the paper. Circuit
modifications (more observables, data re-uploading) are a SEPARATE experiment --
run those in warmstart_pipeline.py.

WHY MULTI-SEED: the from-scratch hybrid is init-fragile (notes.md #4). A single
run is meaningless; the honest comparison is the DISTRIBUTION over seeds.

Run:
    VARIANT=Q2E2 CMP_SEEDS=0,1,2 python baseline_compare.py
    CMP_VARIANTS=Q2E1,Q2E2 python baseline_compare.py

Env (LRs and epochs are independently tunable; the paper's ORIG_LR is NOT tangled
with our method's LRs):
    CMP_VARIANTS   variants to run           (default: VARIANT)
    CMP_SEEDS      seeds                     (default 0,1,2,3,4)
    CMP_METHODS    original,warmstart        (default both)
    ORIG_EPOCHS / ORIG_LR       from-scratch schedule   (default 60 / 1e-5)
    WARM_EPOCHS  / WARM_LR       warm-start phase 1      (default 25 / 1e-3)
    FREEZE_EPOCHS/ QUANTUM_LR    warm-start phase 2      (default 25 / 3e-4)
    FINETUNE_EPOCHS / FINETUNE_LR warm-start phase 3     (default 10 / LR/100)
    ESCAPE_THRESHOLD            "escaped 50%" if best >= (default 0.60)
"""

from __future__ import annotations

import os
import csv
import datetime
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
import torch
import torch.nn as nn
import torch.optim as optim

import warmstart_pipeline as wp


CMP_VARIANTS = [v.strip().upper() for v in
                os.environ.get("CMP_VARIANTS", wp.VARIANT).split(",") if v.strip()]
SEEDS = [int(s) for s in os.environ.get("CMP_SEEDS", "0,1,2,3,4").split(",") if s.strip()]
METHODS = [m.strip().lower() for m in
           os.environ.get("CMP_METHODS", "original,warmstart").split(",") if m.strip()]

# --- ORIGINAL (paper) schedule -- its own lr, no augmentation ---
ORIG_EPOCHS = int(os.environ.get("ORIG_EPOCHS", "60"))
ORIG_LR = float(os.environ.get("ORIG_LR", "0.00001"))        # the paper's exact lr

# --- WARM-START schedule -- three phases, each with its own lr (default to the
# pipeline's, but independently overridable) ---
WARM_EPOCHS = int(os.environ.get("WARM_EPOCHS", "25"))
WARM_LR = float(os.environ.get("WARM_LR", str(wp.LR)))
FREEZE_EPOCHS = int(os.environ.get("FREEZE_EPOCHS", "25"))
FROZEN_LR = float(os.environ.get("QUANTUM_LR", str(wp.QUANTUM_LR)))
FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", "10"))
FT_LR = float(os.environ.get("FINETUNE_LR", str(wp.FINETUNE_LR)))

ESCAPE_THRESHOLD = float(os.environ.get("ESCAPE_THRESHOLD", "0.60"))
# Restore-best BETWEEN phases (warm-start only): at the end of each phase, roll the
# model back to its best-val epoch before the next phase starts -- so frozen
# inherits the best warm-up backbone, and fine-tune inherits the best frozen head.
# Safe to keep on (unlike early stopping it does NOT change epoch count, so the
# curves stay equal-length and averageable). Mirrors warmstart_pipeline's
# RESTORE_BEST. Off (0) = each phase hands over its LAST epoch.
CMP_RESTORE_BEST = os.environ.get("CMP_RESTORE_BEST", "1") != "0"

PAPER_ACC = {"CNN": 90.8, "Q2E1": 92.8, "Q2E2": 95.0, "Q4E1": 93.0, "Q4E2": 93.2}


def _paper_circuit(variant):
    """The paper's exact circuit, pinned -- immune to OBSERVABLES/REUPLOAD env."""
    return wp.create_qnn(variant, obs_mode="single", reupload=1)


# ---------------------------------------------------------------------------
# Clean fixed-epoch training loop -> (best val acc, [(train_loss, val_acc), ...]).
# No lr decay, no early stop -- runs every epoch so curves are full + averageable.
# ---------------------------------------------------------------------------
def _train(model, train_loader, eval_loader, epochs, lr, restore_best=False):
    opt = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best, curve, best_state = -1.0, [], None
    for _ in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(wp.device), yb.to(wp.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg = sum(losses) / max(len(losses), 1)
        acc, _ = wp.evaluate(model, eval_loader)
        curve.append((avg, acc))
        if acc > best:
            best = acc
            if restore_best:                     # snapshot best-val weights
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
    # roll back to the best epoch, so the NEXT phase inherits the best weights
    # (this changes the handed-over model, NOT the recorded curve or epoch count)
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return best, curve


# ---------------------------------------------------------------------------
# The two schedules. Each returns {"best": float, "phases": [(name, curve), ...]}
# where curve is a list of (train_loss, val_acc) per epoch.
# ---------------------------------------------------------------------------
def run_original(seed, variant):
    """Paper's schedule: full hybrid, from scratch, ONE phase, NO augmentation."""
    wp.set_seed(seed); wp.apply_determinism()
    train_loader, eval_loader = wp.make_loaders(augment=False)      # paper used none
    model = wp.QuantumNet(wp.Backbone(), _paper_circuit(variant)).to(wp.device)
    # single phase, so restore_best only sets the final weights (best is reported
    # either way); kept for consistency
    best, curve = _train(model, train_loader, eval_loader, ORIG_EPOCHS, ORIG_LR,
                         restore_best=CMP_RESTORE_BEST)
    return {"best": best, "phases": [("original", curve)]}


def run_warmstart(seed, variant):
    """Our method: classical warm-up -> frozen quantum head -> fine-tune."""
    wp.set_seed(seed); wp.apply_determinism()
    train_loader, eval_loader = wp.make_loaders(augment=wp.AUGMENT)

    backbone = wp.Backbone().to(wp.device)
    # phase 1: restore_best -> the frozen phase inherits the BEST warm-up backbone
    _, warm = _train(wp.ClassicalNet(backbone).to(wp.device),        # phase 1
                     train_loader, eval_loader, WARM_EPOCHS, WARM_LR,
                     restore_best=CMP_RESTORE_BEST)

    wp.set_seed(seed + 1)                                            # head init
    qnet = wp.QuantumNet(backbone, _paper_circuit(variant)).to(wp.device)
    wp.set_backbone_trainable(qnet, False)                          # phase 2: freeze
    # phase 2: restore_best -> fine-tune inherits the BEST frozen head (the frozen
    # phase oscillates, so its last epoch is often NOT its best)
    best_fr, frozen = _train(qnet, train_loader, eval_loader, FREEZE_EPOCHS, FROZEN_LR,
                             restore_best=CMP_RESTORE_BEST)

    wp.set_backbone_trainable(qnet, True)                           # phase 3: unfreeze
    best_ft, fine = _train(qnet, train_loader, eval_loader, FINETUNE_EPOCHS, FT_LR,
                           restore_best=CMP_RESTORE_BEST)

    # "best" is over the QUANTUM phases (frozen + fine-tune); warm-up is classical
    return {"best": max(best_fr, best_ft),
            "phases": [("warmup", warm), ("frozen", frozen), ("finetune", fine)]}


METHOD_FNS = {"original": run_original, "warmstart": run_warmstart}


# ---------------------------------------------------------------------------
# Driver + summary
# ---------------------------------------------------------------------------
def run_all():
    results = {}
    for variant in CMP_VARIANTS:
        results[variant] = {}
        for m in METHODS:
            if m not in METHOD_FNS:
                print(f"  (skipping unknown method {m!r})"); continue
            print(f"\n=== {variant} / {m} ===")
            rows = []
            for s in SEEDS:
                r = METHOD_FNS[m](s, variant)
                escaped = r["best"] >= ESCAPE_THRESHOLD
                rows.append((s, r["best"], escaped, r["phases"]))
                print(f"  seed={s}: best={100 * r['best']:5.2f}%  "
                      f"{'escaped' if escaped else 'STUCK ~50%'}")
            results[variant][m] = rows
    return results


def summarize(rows):
    bests = [b for _, b, _, _ in rows]
    return {"n": len(rows), "escaped": sum(e for _, _, e, _ in rows),
            "mean": statistics.mean(bests),
            "std": statistics.pstdev(bests) if len(bests) > 1 else 0.0,
            "min": min(bests), "max": max(bests)}


def _concat(phases):
    """[(name, curve)] -> (losses, accs, boundaries, spans). boundaries are the
    cumulative epoch indices where a phase ends; spans = (name, start, end)."""
    losses, accs, boundaries, spans, x = [], [], [], [], 0
    for name, curve in phases:
        start = x + 1
        for (l, a) in curve:
            losses.append(l); accs.append(100 * a); x += 1
        spans.append((name, start, x))
        boundaries.append(x)
    return losses, accs, boundaries[:-1], spans          # drop final boundary


def _mean_curve(rows):
    """Mean (loss, acc) over seeds -- all seeds are the same length (no early stop)."""
    L = [np.array(_concat(ph)[0]) for _, _, _, ph in rows]
    A = [np.array(_concat(ph)[1]) for _, _, _, ph in rows]
    return np.mean(L, axis=0), np.mean(A, axis=0)


# ---------------------------------------------------------------------------
# Plots -- SEPARABLE per method, plus a combined figure
# ---------------------------------------------------------------------------
_C = plt.cm.viridis


def _ref_lines(ax, variant, x0=1):
    ax.axhline(50, ls=":", c="0.6", lw=1); ax.text(x0, 51, "chance", fontsize=7, color="0.45")
    if variant in PAPER_ACC:
        ax.axhline(PAPER_ACC[variant], ls="--", c="crimson", lw=1)
        ax.text(x0, PAPER_ACC[variant] + 0.6, f"paper {PAPER_ACC[variant]}%",
                fontsize=7, color="crimson")


def plot_method(variant, method, rows, out_dir):
    """One method alone: train loss + val accuracy, one line per seed."""
    fig, (axL, axA) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (seed, best, _e, ph) in enumerate(rows):
        losses, accs, bounds, _sp = _concat(ph)
        c = _C(i / max(len(rows) - 1, 1))
        xs = range(1, len(losses) + 1)
        axL.plot(xs, losses, color=c, lw=1.3, label=f"seed {seed}")
        axA.plot(xs, accs, color=c, lw=1.3, label=f"seed {seed} ({100 * best:.1f}%)")
    for b in bounds:                                    # phase boundaries (same for all seeds)
        axL.axvline(b + 0.5, ls="--", c="0.5", lw=1); axA.axvline(b + 0.5, ls="--", c="0.5", lw=1)
    for name, a, b in _sp:
        axA.text((a + b) / 2, 42, name, fontsize=7, color="0.35", ha="center")
    axL.set_title("train loss"); axL.set_xlabel("epoch"); axL.set_ylabel("loss"); axL.grid(alpha=.3)
    axA.set_title("val accuracy"); axA.set_xlabel("epoch"); axA.set_ylabel("acc (%)")
    axA.set_ylim(40, 101); axA.grid(alpha=.3); _ref_lines(axA, variant)
    axA.legend(fontsize=7, loc="lower right")
    s = summarize(rows)
    fig.suptitle(f"{variant} / {method}  (escaped {s['escaped']}/{s['n']}, "
                 f"best {100 * s['max']:.1f}%, mean {100 * s['mean']:.1f}+/-{100 * s['std']:.1f})")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{variant}_{method}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_combined(variant, per_method, out_dir):
    """Our full method (warm-up+frozen+fine-tune) spans the whole axis; the paper's
    method is overlaid OFFSET to begin where our quantum (phase-2) training starts
    -- so the first WARM_EPOCHS show only our warm-up, then both are aligned at the
    point each begins training the quantum head (ours warm, theirs from scratch)."""
    if "warmstart" not in per_method:
        return None
    fig, (axL, axA) = plt.subplots(1, 2, figsize=(12, 4.8))

    wsL, wsA = _mean_curve(per_method["warmstart"])
    xw = np.arange(1, len(wsL) + 1)
    axL.plot(xw, wsL, color="#2563eb", lw=2, label="warm-start (ours)")
    axA.plot(xw, wsA, color="#2563eb", lw=2, label="warm-start (ours)")

    # phase boundaries of our method + region labels (warm-up | frozen | fine-tune)
    total = WARM_EPOCHS + FREEZE_EPOCHS + FINETUNE_EPOCHS
    regions = [("warm-up", 0, WARM_EPOCHS),
               ("frozen", WARM_EPOCHS, WARM_EPOCHS + FREEZE_EPOCHS),
               ("fine-tune", WARM_EPOCHS + FREEZE_EPOCHS, total)]
    for b in (WARM_EPOCHS, WARM_EPOCHS + FREEZE_EPOCHS):
        for ax in (axL, axA):
            ax.axvline(b + 0.5, ls="--", c="0.6", lw=1)
    for lbl, a, b in regions:
        axA.text((a + b) / 2, 42, lbl, fontsize=7, ha="center", color="#2563eb")

    if "original" in per_method:
        oL, oA = _mean_curve(per_method["original"])
        xo = np.arange(1, len(oL) + 1) + WARM_EPOCHS     # OFFSET: appears at phase 2
        axL.plot(xo, oL, color="#d97706", lw=2, label="original / paper (from scratch)")
        axA.plot(xo, oA, color="#d97706", lw=2, label="original / paper (from scratch)")

    axL.set_title("mean train loss"); axL.set_xlabel("epoch (cumulative)")
    axL.set_ylabel("loss"); axL.grid(alpha=.3); axL.legend(fontsize=8)
    axA.set_title("mean val accuracy"); axA.set_xlabel("epoch (cumulative)")
    axA.set_ylabel("acc (%)"); axA.set_ylim(40, 101); axA.grid(alpha=.3)
    _ref_lines(axA, variant); axA.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{variant}: our full method vs the paper's (paper overlaid from "
                 f"phase 2; mean over {len(SEEDS)} seeds)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{variant}_both.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    wp.apply_determinism()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", f"compare_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    header = (f"variants={CMP_VARIANTS}  methods={METHODS}  seeds={SEEDS}  "
              f"device={wp.device}\n"
              f"original : {ORIG_EPOCHS} ep @ lr {ORIG_LR} (from scratch, no aug)\n"
              f"warmstart: warmup {WARM_EPOCHS}@{WARM_LR} | frozen {FREEZE_EPOCHS}@{FROZEN_LR} "
              f"| finetune {FINETUNE_EPOCHS}@{FT_LR}  (aug={wp.AUGMENT})\n"
              f"restore_best between phases = {CMP_RESTORE_BEST}  |  "
              f"circuit PINNED to paper's (single observable, reupload=1) for both")
    print(header)

    results = run_all()

    # --- save EVERYTHING: per-epoch curves, per-seed bests, summary, plots ---
    with open(os.path.join(out_dir, "curves.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["variant", "method", "seed", "phase", "epoch", "train_loss", "val_acc"])
        for variant in CMP_VARIANTS:
            for m, rows in results[variant].items():
                for seed, _b, _e, phases in rows:
                    for name, curve in phases:
                        for ep, (loss, acc) in enumerate(curve, 1):
                            w.writerow([variant, m, seed, name, ep, f"{loss:.4f}", f"{acc:.4f}"])
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["variant", "method", "seed", "best_acc", "escaped"])
        for variant in CMP_VARIANTS:
            for m, rows in results[variant].items():
                for seed, best, esc, _ in rows:
                    w.writerow([variant, m, seed, f"{best:.4f}", int(esc)])

    log = [header, ""]
    for variant in CMP_VARIANTS:
        log.append(f"\n===== {variant} =====")
        for m, rows in results[variant].items():
            s = summarize(rows)
            log.append(f"  {m:10s} | escaped {s['escaped']}/{s['n']} | "
                       f"mean {100 * s['mean']:5.2f}% +/- {100 * s['std']:4.2f} | "
                       f"min {100 * s['min']:5.2f}% | best {100 * s['max']:5.2f}%")
            plot_method(variant, m, rows, out_dir)                # SEPARABLE figure
        if variant in PAPER_ACC:
            log.append(f"  {'paper':10s} | single reported run: {PAPER_ACC[variant]}%")
        plot_combined(variant, results[variant], out_dir)        # combined figure

    report = "\n".join(log)
    print("\n" + "=" * 68 + "\n" + report)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsaved to: {os.path.abspath(out_dir)}")
    print("  summary.txt | results.csv (best/seed) | curves.csv (per-epoch) | "
          "<variant>_<method>.png (each alone) | <variant>_both.png (together)")
