"""
The power of one clean qubit in supervised machine learning (arXiv:2210.09275).

Modernized for Qiskit 2.x (tested on qiskit==2.4.2, qiskit-machine-learning==0.9.0,
scikit-learn==1.9.0, numpy==2.x).

WHAT THIS DOES
--------------
Implements the DQC1 quantum kernel for binary classification on the "ad_hoc"
dataset, then trains a classical SVM on the precomputed kernel.

  qubit 0        -> "clean" control qubit  (Hadamard test probe)
  qubits 1, 2    -> target qubits (feature register)
  qubits 3, 4    -> ancillas, used to put the targets into a maximally mixed
                    state via Bell pairs (Fig. 4a of the paper)

For each pair (x_i, x_j) we build U = u^l(x_i) . u^l(x_j)^dagger with l = 2
(the feature map of Eqs. 20-21). The off-diagonal element rho_01 of the control
qubit's reduced density matrix equals (1/2) * K(x_i, x_j)  (Eq. 4 / Eq. 18),
so the kernel value is  K = 2 * |rho_01|.

MIGRATION NOTE (vs. the original qiskit==0.34.1 code)
-----------------------------------------------------
The original relied on qiskit.ignis state tomography + qiskit.execute + Aer,
all of which were removed in Qiskit 1.0. On a *noiseless simulator* the control
qubit's density matrix can be computed exactly, with no shots and no tomography
fitter, using qiskit.quantum_info.DensityMatrix + partial_trace. That is what we
do here (this reproduces the paper's simulation results, e.g. 100% accuracy).
To study hardware noise (Figs. 9-11) instead, swap `dqc1_kernel_value` for a
shot-based estimate using qiskit-experiments StateTomography + a Sampler.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace

from sklearn import svm
from sklearn.metrics import accuracy_score

from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.utils import algorithm_globals


# ---------------------------------------------------------------------------
# Dataset / dimensionality
#
# N_FEATURES = number of input features = number of target qubits. Everything
# below is generalized to this value; set it to 2, 3, ... to change the kernel
# dimensionality. The total qubit count of each DQC1 circuit is 2*N_FEATURES + 1
# (1 control + N_FEATURES targets + N_FEATURES mixing ancillas), so runtime and
# memory grow quickly -- n=2 -> 5 qubits, n=3 -> 7 qubits, n=4 -> 9 qubits, ...
#
# The built-in ad_hoc_data generator only supports n in {2, 3}; for N_FEATURES
# >= 4 the loader below automatically falls back to a scalable sklearn dataset.
# ad_hoc_data is randomly generated; seed it so runs are reproducible.
# ---------------------------------------------------------------------------
# Set N_FEATURES to change the kernel dimensionality (the whole pipeline is
# generalized to it). CAVEAT: the DQC1 kernel value is |tr(U)|/2^n, which
# concentrates toward 0 as n grows (exponential concentration, discussed in the
# paper's conclusion). Empirically n=2 -> ~100% on ad_hoc, but n=3 already
# collapses to chance (~0.5) even though the data is separable by a fidelity
# kernel. So n=2 is the meaningful DQC1 setting; use n=3 only to *observe* the
# concentration effect. For strong higher-dim accuracy use the fidelity kernel
# (see eval_harness.py).
algorithm_globals.random_seed = 12345
N_FEATURES = 2          # number of features / target qubits (ad_hoc supports 2 or 3)
LAYERS = 2              # l, number of feature-map repetitions

if N_FEATURES <= 3:
    train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
        training_size=20,
        test_size=5,
        n=N_FEATURES,
        gap=0.3,
        plot_data=False,
        one_hot=False,
        include_sample_total=True,
    )
else:
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import MinMaxScaler
    _X, _y = make_classification(
        n_samples=50, n_features=N_FEATURES, n_informative=N_FEATURES,
        n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=12345,
    )
    _X = MinMaxScaler((0, 2 * np.pi)).fit_transform(_X)   # scale into gate angles
    train_features, train_labels = _X[:40], _y[:40]
    test_features, test_labels = _X[40:], _y[40:]

# 20 samples per label -> 40 training points, 5 per label -> 10 test points.
n_train = len(train_features)   # 40
n_test = len(test_features)     # 10


# ---------------------------------------------------------------------------
# DQC1 circuit and kernel value  (generalized to N_FEATURES target qubits)
#
# Qubit layout for n = N_FEATURES:
#   qubit 0            -> control ("clean") qubit
#   qubits 1 .. n      -> target qubits (feature register)
#   qubits n+1 .. 2n   -> ancillas that maximally-mix the targets (traced out)
#
# The feature map is the Havlicek ZZ map (Eq. 21): H on every target, a
# single-qubit Z rotation per feature, and a ZZ interaction for every pair of
# features -- all controlled by the clean qubit (the Hadamard-test structure).
# ---------------------------------------------------------------------------
TARGETS = list(range(1, N_FEATURES + 1))


def _u_layer(qc, x):
    """One controlled layer of the feature map u(x) on the target qubits."""
    for t in TARGETS:                                    # H^{\otimes n}
        qc.ch(0, t)
    for feat, t in enumerate(TARGETS):                   # single-qubit Z(2 x_i)
        qc.crz(2 * x[feat], 0, t)
    for a in range(len(TARGETS)):                        # ZZ over every pair (i,j)
        for b in range(a + 1, len(TARGETS)):
            ta, tb = TARGETS[a], TARGETS[b]
            qc.ccx(0, ta, tb)
            qc.crz(2 * (np.pi - x[a]) * (np.pi - x[b]), 0, tb)
            qc.ccx(0, ta, tb)


def _u_dagger_layer(qc, x):
    """Inverse of _u_layer: same gates, reversed order, negated angles."""
    for a in reversed(range(len(TARGETS))):
        for b in reversed(range(a + 1, len(TARGETS))):
            ta, tb = TARGETS[a], TARGETS[b]
            qc.ccx(0, ta, tb)
            qc.crz(-2 * (np.pi - x[a]) * (np.pi - x[b]), 0, tb)
            qc.ccx(0, ta, tb)
    for feat in reversed(range(len(TARGETS))):
        qc.crz(-2 * x[feat], 0, TARGETS[feat])
    for t in reversed(TARGETS):
        qc.ch(0, t)


def dqc1_circuit(x_i, x_j):
    """DQC1 circuit for the feature-vector pair (x_i, x_j) with l = LAYERS."""
    qc = QuantumCircuit(2 * N_FEATURES + 1)
    qc.h(0)                              # control qubit -> |+>
    for t in TARGETS:                    # targets -> maximally mixed via Bell pairs
        qc.h(t)
        qc.cx(t, t + N_FEATURES)         # entangle with ancilla, later traced out
    for _ in range(LAYERS):              # U = u^l(x_i) . u^l(x_j)^dagger
        _u_layer(qc, x_i)
    for _ in range(LAYERS):
        _u_dagger_layer(qc, x_j)
    return qc


def dqc1_kernel_value(x_i, x_j):
    """Exact kernel value K(x_i, x_j) = 2*|rho_01| of the control qubit."""
    dm = DensityMatrix(dqc1_circuit(x_i, x_j))
    rho_c = partial_trace(dm, list(range(1, 2 * N_FEATURES + 1)))  # keep qubit 0
    return 2 * abs(rho_c.data[0, 1])


# ---------------------------------------------------------------------------
# Training Gram matrix (40 x 40)
# ---------------------------------------------------------------------------
t = time.time()
k = np.zeros((n_train, n_train))
for i in range(n_train):
    for j in range(n_train):
        k[i, j] = dqc1_kernel_value(train_features[i], train_features[j])
print("Training kernel time:", time.time() - t)

# Diagonal should be ~1 in the noiseless case (K(x, x) = 1).
d = [k[i, i] for i in range(n_train)]
print("Diagonal (should be ~1):", np.round(d, 3))


# ---------------------------------------------------------------------------
# Test Gram matrix (10 x 40)
# ---------------------------------------------------------------------------
t = time.time()
kt = np.zeros((n_test, n_train))
for i in range(n_test):
    for j in range(n_train):
        kt[i, j] = dqc1_kernel_value(test_features[i], train_features[j])
print("Test kernel time:", time.time() - t)


# ---------------------------------------------------------------------------
# Classical SVM on the precomputed quantum kernel
# ---------------------------------------------------------------------------
clf = svm.SVC(kernel="precomputed")
clf.fit(k, train_labels)
Y_predict = clf.predict(kt)

print("Predicted:", Y_predict)
print("True     :", test_labels)
print("Accuracy :", accuracy_score(Y_predict, test_labels))


# ---------------------------------------------------------------------------
# Plot: training / prediction scatter
# NOTE: the original code drew an RdBu background from `adhoc_total`. In
# qiskit-machine-learning 0.9.0 `include_sample_total` returns an integer count,
# not the label grid, so that decorative background is no longer available and
# is omitted here. The scatter below still shows the classification result.
# ---------------------------------------------------------------------------
plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.scatter(
    train_features[train_labels == 0, 0], train_features[train_labels == 0, 1],
    marker="s", facecolors="w", edgecolors="b", label="A train",
)
plt.scatter(
    train_features[train_labels == 1, 0], train_features[train_labels == 1, 1],
    marker="o", facecolors="w", edgecolors="r", label="B train",
)
plt.scatter(
    test_features[Y_predict == 0, 0], test_features[Y_predict == 0, 1],
    marker="s", facecolors="b", edgecolors="w", label="A test",
)
plt.scatter(
    test_features[Y_predict == 1, 0], test_features[Y_predict == 1, 1],
    marker="o", facecolors="r", edgecolors="w", label="B test",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.title("Ad hoc dataset")
plt.show()


# ---------------------------------------------------------------------------
# Kernel matrix heatmap
# ---------------------------------------------------------------------------
plt.figure(figsize=(5, 5))
plt.imshow(np.asarray(k), interpolation="nearest", origin="upper")
plt.title("Analytical Kernel Matrix")
plt.colorbar()
plt.show()


# ---------------------------------------------------------------------------
# Coherence consumption, Eq. (19):  Delta C = H2((1 - K) / 2)
# ---------------------------------------------------------------------------
x1 = np.abs(1 - k) / 2
# guard the endpoints so log2(0) does not produce NaNs
x1 = np.clip(x1, 1e-12, 1 - 1e-12)
h1 = -x1 * np.log2(x1) - (1 - x1) * np.log2(1 - x1)
print("Max coherence consumption:", np.max(h1))

h2 = np.reshape(h1, (n_train, n_train))
plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.imshow(
    np.asarray(h2), interpolation="nearest", origin="upper",
    extent=[0, 2 * np.pi, 0, 2 * np.pi], vmin=0, vmax=1,
)
plt.colorbar()
plt.title("Coherence consumption")
plt.show()


# ---------------------------------------------------------------------------
# Geometric discord, Eq. (12) with alpha = 1:  D_G = (1 / 2^(n+2)) (1 - K)
# (for n = 2 the prefactor is 1/16, matching the original code)
# ---------------------------------------------------------------------------
discord_scale = 1 / 2 ** (N_FEATURES + 2)
dis = discord_scale * (1 - k)
print("Max discord:", np.max(dis))

plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.imshow(
    np.asarray(dis), interpolation="nearest", origin="upper",
    extent=[0, 2 * np.pi, 0, 2 * np.pi], vmin=0, vmax=discord_scale,
)
plt.colorbar()
plt.title("Geometric discord")
plt.show()
