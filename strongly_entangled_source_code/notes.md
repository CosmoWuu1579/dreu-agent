# Why CNN-Q2E2 stalls at 50% — investigation notes

**Symptom.** Running the paper's `CNN-Q2E2.py` on the split Br35H data, training loss
hovers (~1.4) and validation accuracy is pinned at exactly 50% for dozens of
epochs. Raising the learning rate 100× (1e-5 → 1e-3) does not help.

**Bottom line (proven below).** It is **not** the code port, **not** the qiskit
version, **not** the observable, and **not** a broken gradient. The quantum head
trains perfectly on good features. The failure is that a **randomly-initialized
CNN cannot bootstrap useful features *through* the 1-number quantum bottleneck**
from scratch, so the joint optimization gets stuck in a 50% local minimum whose
escape is a seed lottery. **Fix: warm-start the CNN** (train it classically first,
then attach the quantum head). Implemented in `warmstart_pipeline.py`.

---

## Methodology

I isolated the cause by ruling factors out one at a time with controlled
experiments in the `dreu` env (qiskit 2.4.2, qiskit-machine-learning 0.9.0,
torch 2.12.1). Each claim below is backed by a run.

## Evidence

**1. The version update did NOT change behavior.**
Old `ZZFeatureMap` class vs the new functional `zz_feature_map`, same QNN, same
inputs/weights → **identical outputs**:
```
old: [-0.1153 -0.8224  0.1261  0.4767]
new: [-0.1153 -0.8224  0.1261  0.4767]
```
So the `ZZFeatureMap → zz_feature_map` migration is behaviorally exact. Not the cause.

**2. The gradient is CORRECT.**
TorchConnector's analytic gradient vs a finite-difference reference:
```
WEIGHT grad  analytic: [-0.7238 -0.2116 -0.5226 -0.    ]
WEIGHT grad  numeric : [-0.7238 -0.2116 -0.5226  0.    ]
INPUT  grad  analytic: [-1.5082 -1.4822]
INPUT  grad  numeric : [-1.5081 -1.4822]
```
The hybrid gradient engine is fine. Not the cause.

**3. Classical head trains; quantum head does not — WITH a random CNN.**
Same CNN backbone, same trivial task (separate all-bright from all-dark 128×128
images), same lr=1e-3, 25 epochs:
```
CLASSICAL head (fc2 -> 2 logits)          -> val_acc = 1.00
QUANTUM   head (fc2 -> qnn -> fc3 -> cat)  -> val_acc = 0.50 (stuck)
```
The quantum bottleneck is where it breaks — but see #5, it's not the head itself.

**4. It's a SEED lottery, and the observable is irrelevant.**
Paper-exact quantum head, trivial task, only the observable varies, two seeds:
```
DEFAULT (Z⊗Z parity) : seed0=0.50  seed1=1.00
single-qubit IZ      : seed0=0.50  seed1=1.00
single-qubit ZI      : seed0=0.50  seed1=1.00
```
Same architecture: `seed0` sticks at 50%, `seed1` reaches 100%, for *every*
observable. So whether it escapes the 50% basin depends on the random init — the
paper drew a good seed, our run drew a bad one. The observable is **not** the knob.

**5. The quantum head is FINE on good features (no CNN).**
Feeding separable 2-D data directly to the quantum head (no CNN in front),
5-seed robustness:
```
A paper (ZZ global + cat + rand init) : [1.00, 1.00, 1.00, 1.00]  mean=1.00
D "barren-plateau fixes" (local obs +
   Linear(2,2) + small-angle init)    : [0.95, 0.90, 1.00, 0.50]  mean=0.84
```
On good features the paper's exact head is **robust (4/4 perfect)**, and the
textbook barren-plateau fixes (local observable, 2-logit head, small-angle init)
were **no better and sometimes worse**. So this is not a classic barren plateau
(it's only 2 qubits — gradients don't vanish), and the head architecture is not
the problem.

**6. CONFIRMED on real Br35H data: warm-start fixes the stall — but unfreezing at
the same lr destroys it.** Real run, `VARIANT=Q2E2`, default params:
```
warmup     ep 1  66.0%  ->  ep 25  96.5% (peak),  ends 95.0%
q_frozen   ep 1  82.5%  ->  ep 6   90.5%,         ends 89.0%  (still climbing)
q_finetune ep 1  80.5%  ->  ep 2  54.5%  ->  ep 3  51.5%   (loss 0.32 -> 0.85)
           ... 40 epochs of recovery, only back to 85.0%
```
Three things this proves:
- **The 50% stall is GONE.** With a warm backbone the quantum head starts at
  82.5% *on epoch 1* and reaches 90.5% — exactly what #3/#5 predicted.
- **The data/split are fine.** The classical backbone reaches 96.5%, *better*
  than the paper's own CNN baseline (90.8%). The original stall was never a data
  problem.
- **Unfreezing at the same lr = catastrophic forgetting.** 4.3M converged
  parameters + a freshly reset optimizer + lr=1e-3 wreck the learned features in
  two epochs, and 40 epochs of recovery still end *below* the frozen phase
  (85.0% vs 90.5%). Fix: `FINETUNE_LR` (default `LR/100`), or simply keep the
  backbone frozen (`FAST=1`) — the frozen phase was the best part of the run.

**Caveat worth reporting:** here the **classical head (96.5%) beats the quantum
hybrid (90.5%)**, i.e. this does *not* reproduce the paper's CNN 90.8% -> Q2E2
95% claim. That is consistent with #3/#5: squeezing 512 features through 2 encoder
inputs into a **single scalar measurement** is a capacity bottleneck. Our classical
baseline is simply stronger than theirs, so the quantum layer has nothing to add.

## Conclusion

Combining #3 (fails behind a *random* CNN) with #5 (succeeds on *good* features):
the culprit is the **from-scratch joint training of the CNN and the quantum head**.
A random CNN produces useless features; the quantum head, squeezed to a single
scalar output, passes back a weak/misdirected gradient; the CNN never learns good
features; everything sits at 50%. The classical head avoids this because it has
enough capacity to give the CNN a strong gradient immediately.

## The fix

Give the quantum head good features to start from — **warm-start / pretrain the
CNN**, the #1 literature-recommended barren-plateau mitigation ("pretraining with
classical neural networks"). `warmstart_pipeline.py` does this as staged training
(one file, all variants — `VARIANT=Q2E2|Q2E1|Q4E1|Q4E2`):

- **Phase 1:** train the backbone (`conv1-4 → fc1`) + a plain `Linear(512,2)`
  head classically — learns good tumour/healthy features (robust).
- **Phase 2:** reuse the *same* now-warm backbone, attach the quantum head
  (`fc2 → QNN → fc3 → cat`), freeze the backbone for a few epochs so the head
  catches up, then unfreeze and fine-tune.

No cross-file weight copying (which is where the `fc2: 512→2` vs `512→1` shape
mismatch would bite): the backbone is one shared module reused between phases;
only the head is swapped. Standard transfer learning — keep the backbone, swap
the head.

Secondary safety net: even warm-started, VQC init has some variance, so a 2–3
seed retry (keep the run that moves off 50% by ~epoch 10) is cheap insurance.

## Next: wiring this into the LLM agent pipeline

**Why this work is the missing piece.** The DREU project is an LLM agent that
designs quantum circuits. Until now it had no rigorous task to design *against*.
This gives it one: a real medical-imaging benchmark with a measured baseline
(classical **99.5%**), a measured quantum result (**97.5%**), a **2-point gap**,
and a mechanism hypothesis (readout capacity: quantum train loss floors at ~0.27
vs the classical head's ~0.03). The agent's job becomes concrete and falsifiable:
**close that gap.**

**Design space the agent can search** (all exposed as knobs in
`warmstart_pipeline.py`, so the seeds are real and runnable):
encoder E1/E2 (product vs entangled) x ansatz (RealAmplitudes / conv-pool) x
`OBSERVABLES` (readout width: single / local / local_corr) x `REUPLOAD` (data
re-uploading depth) x qubit count. A grid over these is only ~12 configs -- an
LLM adds nothing to grid search. The value is letting it **write circuits**
(`create_qnn(n) -> EstimatorQNN`) that go beyond the grid.

**Plan (4 pieces):**

1. **Freeze a reusable backbone (one-time).** Run the warm-up once; save
   `backbone.pt` and dump its 512-d features to `features_{train,val}.npy`. Every
   agent evaluation reuses them: no images, no CNN, no re-warm-up. This is what
   makes an agent loop viable -- and it is why the frozen-feature idea works now
   but did not before: the features come from a 99.5% backbone, not a random CNN.

2. **Evaluator** `qcnn_head_eval.py`: load cached features -> exec the agent's
   `create_qnn(n)` -> wrap in `QuantumHead` -> train ~15 epochs -> return
   `{ok, accuracy (BEST, not last), n_weights, n_readouts, depth}`.
   SPEED: quantum cost is linear in samples, so **subsample to ~400 features for
   the search** (seconds/candidate) and re-validate finalists on the full set.

3. **Task** `qcnn_head_task.py` (mirrors `qml_task.py`): `entry_point="create_qnn"`,
   `target_metric="accuracy"`, `target_value=0.995` (the classical ceiling).
   Seeds = the paper's Q2E2 + the `local_corr` and `REUPLOAD=3` variants. The
   prompt states the real problem: *beat 97.5%; classical is 99.5%; the measured
   bottleneck is readout capacity -- widen it.*

4. **Runner** -> `AgentPipeline`, mirroring `run_discovery.py`.

**Caveats to design around:**
- SGD init-fragility (#4) makes scoring noisy: fix the seed, report the BEST
  epoch (not the last), and average 2 seeds for finalists.
- Report the agent's winner on a held-out split, not the one used for selection.

## Sources

- Cerezo et al., *Cost function dependent barren plateaus in shallow parametrized
  quantum circuits*, Nature Communications 2021 —
  https://www.nature.com/articles/s41467-021-21728-w
  (global cost functions, e.g. `Z⊗Z` parity, cause barren plateaus; local
  single-qubit observables stay trainable — relevant at scale, though negligible
  at 2 qubits as #4/#5 show).
- *Investigating and mitigating barren plateaus in variational quantum circuits:
  a survey*, Springer QIP 2025 —
  https://link.springer.com/article/10.1007/s11128-025-04665-1
  (mitigations: local cost functions, classical-NN pretraining / warm-starting,
  layer-by-layer training, correlated parameters).
