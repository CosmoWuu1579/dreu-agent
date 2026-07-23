# ============================================================
# DQC1 Entropy Clustering 
# ------------------------------------------------------------
# Panels per row (side-by-side, 3 columns):
#   1) DQC1 FULL        (DQC1 normalized-trace kernel + DQC1 purity ΔH + reduction + K*)
#   2) Parzen FULL      (Gaussian kernel, init + ΔH + reduction + K*)
#   3) K-Means baseline (K = #GT classes)
#
# Feature maps for DQC1 (env FEATURE_MAP): zz | zdiag | rxryrz | pauli | karimi
# Base quantum kernel here: DQC1 normalized-trace kernel
# ============================================================
#
# DATASET SELECTION NOTE 

#
# 1) Synthetic:
#    - spirals
#    - Rings with different thickness (factor=0.2)
#    - Moons
#    - Circles
#    - Anisotropic blobs 

import os, math, time, warnings
from typing import List, Tuple, Optional, Dict
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_moons, make_circles, make_blobs, fetch_openml
from sklearn.metrics import pairwise_distances
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- Controls (GLOBAL KNOBS) --------------------
# SMALL-N defaults tuned for DQC1 divergence experiments
INIT_CLUSTERS = int(os.environ.get("INIT_CLUSTERS", 8))   
MIN_CLUSTERS  = int(os.environ.get("MIN_CLUSTERS", 3))
N_INIT        = int(os.environ.get("N_INIT", 2))           
REPEATS       = int(os.environ.get("REPEATS", 2))
RANDOM_STATE  = int(os.environ.get("RANDOM_STATE", 0))
FEATURE_MAP   = os.environ.get("FEATURE_MAP", "karimi").strip().lower()  # zz|zdiag|rxryrz|pauli|karimi
LAYERS        = int(os.environ.get("DQC1_LAYERS", 2))
DATA_NOISE    = float(os.environ.get("DATA_NOISE", 0))
print(f"[CONFIG] DATA_NOISE = {DATA_NOISE}")


# ---- Global profiling accumulators (for end-of-run summary) ----
PROFILE_FABLE_PURITY = bool(int(os.environ.get("DQC1_PROFILE_FABLE_PURITY", 1)))

PROFILE_STATS = {
    "fable_time": 0.0,
    "fable_calls": 0,
    "purity_time": 0.0,
    "purity_calls": 0,
}




# Sigma grid for Parzen kernels
PARZEN_SIGMA_GRID = [float(s) for s in os.environ.get("PARZEN_SIGMA_GRID", "0.06, 0.08, 0.10, 0.12").split(",")]



# Rank for kernel compression before DQC1 purity
DQC1_COMP_RANK = int(os.environ.get("DQC1_COMP_RANK", 8))

DISABLE_TIMERS = int(os.environ.get("DISABLE_TIMERS", 0))

# ---- File naming / labels ----
KERNEL_NAME = os.environ.get("KERNEL_NAME", "DQC1 Normalized-Trace Kernel")
TAG = os.environ.get("TAG", f"dqc1_norm_{FEATURE_MAP}_L{LAYERS}")

# Global workers knob (used as default for all DQC1 workers)
DQC1_GLOBAL_WORKERS = int(os.environ.get("DQC1_GLOBAL_WORKERS", 2))

# Keep per-stage knobs but let them default to the global one
DQC1_KERNEL_WORKERS = int(os.environ.get("DQC1_KERNEL_WORKERS", DQC1_GLOBAL_WORKERS))
DQC1_PURITY_WORKERS = int(os.environ.get("DQC1_PURITY_WORKERS", DQC1_GLOBAL_WORKERS))

# Separate global shot knobs (kernel vs purity can differ)
DQC1_KERNEL_SHOTS = int(os.environ.get("DQC1_KERNEL_SHOTS", 128))
DQC1_PURITY_SHOTS = int(os.environ.get("DQC1_PURITY_SHOTS", 128))

# Exact vs sampled modes
#   DQC1_EXACT_KERNEL = 1 → use statevector for kernel (shots ignored)
#   DQC1_EXACT_KERNEL = 0 → use sampled MPS / non-exact backend (uses DQC1_KERNEL_SHOTS)
#   DQC1_EXACT_PURITY = 1 → use statevector for purity (shots ignored)
#   DQC1_EXACT_PURITY = 0 → use sampled DQC1 purity (uses DQC1_PURITY_SHOTS)
DQC1_EXACT_KERNEL = int(os.environ.get("DQC1_EXACT_KERNEL", 1))
DQC1_EXACT_PURITY = int(os.environ.get("DQC1_EXACT_PURITY", 1))

# synthetic datasets to use (by name from toy_sets)

# available:
#   ['spirals', 'rings_factor02','moons', 'circles', 'anistropic_blobs']
SYN_DATASETS = os.environ.get(
    "SYN_DATASETS",
    "spirals"  # default
)

# -------------------- Utilities --------------------
def zero_mean_to_unit_interval(X):
    eps = 1e-12
    Xc = X - X.mean(axis=0, keepdims=True)
    m  = np.max(np.abs(Xc), axis=0, keepdims=True)
    m  = np.where(m < eps, 1.0, m)
    return Xc / m

def first_large_jump_K(entropy_log):
    Ks, Bs = zip(*entropy_log)
    Ks = np.array(Ks); Bs = np.array(Bs)
    if len(Bs) < 2:
        return int(Ks[-1])
    dB = np.diff(Bs)
    idx = int(np.argmax(dB))
    return int(Ks[idx])

def eval_scores(X, y_true, labels):
    valid = labels >= 0
    Xv = X[valid]; lv = labels[valid]
    K = len(np.unique(lv))
    if K > 1:
        sil = silhouette_score(Xv, lv)
        dbi = davies_bouldin_score(Xv, lv)
        ch  = calinski_harabasz_score(Xv, lv)
    else:
        sil = dbi = ch = np.nan
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return dict(ARI=ari, NMI=nmi, Sil=sil, DBI=dbi, CH=ch, K=K)

def print_scores(title, scores):
    print(f"{title:>24} | K={scores['K']:>2}  "
          f"ARI={scores['ARI']:.3f}  NMI={scores['NMI']:.3f}  "
          f"Sil={scores['Sil']:.3f}  DBI={scores['DBI']:.3f}  CH={scores['CH']:.1f}")

def labels_from_clusters(N, clusters):
    lab = -np.ones(N, dtype=int)
    for k, c in enumerate(clusters):
        lab[np.asarray(c, int)] = k
    return lab

# ============================================================
# ------------------------ DQC1 CORE -------------------------
# ============================================================
import torch as t
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE  = t.float64

def tnp(x_t: t.Tensor) -> np.ndarray:
    return x_t.detach().cpu().numpy()

def npt(x_np: np.ndarray) -> t.Tensor:
    return t.as_tensor(x_np, dtype=DTYPE, device=DEVICE)

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if not DISABLE_TIMERS:
            print(f"[TIMER] {self.name}…", flush=True)
        self.t0 = time.time()
        return self

    def __exit__(self, *_):
        if not DISABLE_TIMERS:
            print(f"[TIMER] {self.name} done in {time.time()-self.t0:.2f}s", flush=True)

def zero_mean_to_unit_interval_t(X: t.Tensor) -> t.Tensor:
    eps = t.tensor(1e-12, device=DEVICE, dtype=DTYPE)
    Xc  = X - X.mean(dim=0, keepdim=True)
    m   = t.max(t.abs(Xc), dim=0, keepdim=True).values
    m   = t.where(m < eps, t.tensor(1.0, device=DEVICE, dtype=DTYPE), m)
    return Xc / m

def make_backend(exact: bool = True, gpu_wanted: bool = True, precision: str = "single"):
    method  = "statevector" if exact else "matrix_product_state"
    backend = AerSimulator(method=method)
    opts = {"blocking_enable": True, "max_parallel_experiments": 1, "max_parallel_threads": None}
    try: opts["precision"] = precision
    except Exception: pass
    opts["device"] = "GPU" if (gpu_wanted and exact) else "CPU"
    try: backend.set_options(**opts)
    except Exception: pass
    try:
        print(f"[Aer] device={backend.options.get('device')} "
              f"precision={backend.options.get('precision')} method={method}")
    except Exception:
        print(f"[Aer] method={method}")
    return backend

def _run_with_cpu_fallback(backend, tcircs, shots: Optional[int], exact: bool):
    try:
        return backend.run(tcircs, shots=None if exact else shots).result()
    except Exception:
        be_cpu = make_backend(exact=exact, gpu_wanted=False)
        return be_cpu.run(tcircs, shots=None if exact else shots).result()

# ---------- Feature maps ----------
def karimi_feature_map(x: np.ndarray) -> QuantumCircuit:
    x = np.asarray(x, dtype=float)
    n = len(x)
    qc = QuantumCircuit(n, name="U_karimi")
    qc.h(range(n))
    for i in range(n):
        qc.rz(2 * np.pi * x[i], i)
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                qc.rzz(2 * np.pi * x[i] * x[j], i, j)
    return qc

def fm_zz(x: np.ndarray, layers: int = 2) -> QuantumCircuit:
    x = np.asarray(x, float); n = len(x)
    qc = QuantumCircuit(n, name="U_zz")
    qc.h(range(n))
    for L in range(layers):
        for q in range(n):
            qc.rz(2*np.pi*x[q], q)
        for i in range(n):
            for j in range(i+1, n):
                qc.rzz(2*np.pi*x[i]*x[j], i, j)
    return qc

def fm_zdiag(x: np.ndarray, layers: int = 2) -> QuantumCircuit:
    x = np.asarray(x, float); n = len(x)
    qc = QuantumCircuit(n, name="U_zdiag")
    for L in range(layers):
        for q in range(n): qc.rz(2*np.pi*x[q], q)
        for q in range(0, n-1, 2): qc.cz(q, q+1)
        for q in range(1, n-1, 2): qc.cz(q, q+1)
    return qc

def fm_rxryrz(x: np.ndarray, layers: int = 2) -> QuantumCircuit:
    x = np.asarray(x, float); n = len(x)
    qc = QuantumCircuit(n, name="U_rxryrz")
    for L in range(layers):
        for q in range(n):
            a = x[q]
            qc.rx(2*np.pi*(a+0.1*L), q)
            qc.ry(2*np.pi*(a*a+0.2*L), q)
            qc.rz(2*np.pi*(0.3*a+0.05*L), q)
        for q in range(n-1): qc.cx(q, q+1)
    return qc

def fm_pauli(x: np.ndarray, layers: int = 2) -> QuantumCircuit:
    x = np.asarray(x, float); n = len(x)
    qc = QuantumCircuit(n, name="U_pauli")
    for L in range(layers):
        for q in range(n): qc.rz(2*np.pi*(0.6*x[q] + 0.2*(x[q]**3)), q)
        for i in range(n):
            for j in range(i+1, n):
                qc.rzz(2*np.pi*(0.4*x[i]*x[j] + 0.1*(x[i]**2 + x[j]**2)), i, j)
    return qc

def build_feature_map(x: np.ndarray, kind: str, layers: int) -> QuantumCircuit:
    k = (kind or "karimi").strip().lower()
    if k == "karimi": return karimi_feature_map(x)
    if k == "zz":     return fm_zz(x, layers)
    if k == "zdiag":  return fm_zdiag(x, layers)
    if k == "rxryrz": return fm_rxryrz(x, layers)
    if k == "pauli":  return fm_pauli(x, layers)
    return karimi_feature_map(x)

def _prep_bell_pairs(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(2 * n, name="prep_mm_bell")
    for i in range(n):
        qc.h(i); qc.cx(i, n + i)
    return qc

def _p0_minus_p1_from_statevector(qc_no_meas: QuantumCircuit, probe_index: int = 0) -> float:
    psi = Statevector.from_instruction(qc_no_meas)
    probs = psi.probabilities_dict()
    z = 0.0
    for bitstr, p in probs.items():
        probe_bit = bitstr[-1 - probe_index]
        z += p if probe_bit == '0' else -p
    return float(z)




def _dqc1_expectation_for_basis(
    U_rel: QuantumCircuit,
    basis: str = "x",
    shots: int = 128,
    exact: bool = True,
) -> float:

    basis = basis.lower()
    if basis not in ("x", "y"):
        raise ValueError(f"Unsupported basis '{basis}' (use 'x' or 'y').")

    n = U_rel.num_qubits
    qc = QuantumCircuit(1 + 2 * n, 1)

    # Prepare maximally mixed targets via Bell pairs
    qc.compose(_prep_bell_pairs(n), qubits=range(1, 1 + 2 * n), inplace=True)

    probe = 0
    targets = list(range(1, 1 + n))

    if basis == "x":
        # Standard DQC1: H - CU - H
        qc.h(probe)
        qc.append(U_rel.control(1), qargs=[probe] + targets)
        qc.h(probe)

    elif basis == "y":
        # Imaginary-part variant: S† - H - CU - H - S
        qc.sdg(probe)
        qc.h(probe)
        qc.append(U_rel.control(1), qargs=[probe] + targets)
        qc.h(probe)
        qc.s(probe)

    if exact:
        # Expectation of Z on the probe (p0 - p1)
        return _p0_minus_p1_from_statevector(
            qc.remove_final_measurements(inplace=False),
            probe_index=probe,
        )

    backend = make_backend(exact=False, gpu_wanted=True)
    qc.measure(probe, 0)
    tqc = transpile(qc, backend, optimization_level=0)
    res = _run_with_cpu_fallback(backend, tqc, shots=shots, exact=False)
    cnts = res.get_counts()
    total = max(1, sum(cnts.values()))
    p0 = cnts.get("0", 0) / total
    p1 = cnts.get("1", 0) / total
    return p0 - p1




def dqc1_kernel_value(
    xi: np.ndarray,
    xj: np.ndarray,
    shots: int = 128,
    exact: bool = True
) -> float:
    """
    DQC1 kernel value following Karimi–Ghobadi:
        K(x, x') ∝ |tr(U_rel)| / 2^n

    Implementation:
      • Build U_rel = U(x_i) U(x_j)^\dagger from the chosen feature map.
      • Use two DQC1 Hadamard tests:
          - basis='x' → Re[tr(U_rel)/2^n]
          - basis='y' → ±Im[tr(U_rel)/2^n]
      • Return sqrt(Re^2 + Im^2), i.e. |tr(U_rel)/2^n|.
    """
    Uxi = build_feature_map(xi, FEATURE_MAP, LAYERS)
    Uxj = build_feature_map(xj, FEATURE_MAP, LAYERS)
    U_rel = Uxi.compose(Uxj.inverse())

    # Real and imaginary parts (normalized by 2^n already via DQC1)
    re = _dqc1_expectation_for_basis(U_rel, basis="x", shots=shots, exact=exact)
    im = _dqc1_expectation_for_basis(U_rel, basis="y", shots=shots, exact=exact)

    # Magnitude of the complex trace part: |tr(U_rel)| / 2^n
    val = math.sqrt(re * re + im * im)

    # Clip into [0, 1]; diagonals are later forced to exactly 1.0 anyway.
    return float(np.clip(val, 0.0, 1.0))


def _make_pairs(N: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(N) for j in range(i, N)]

def build_dqc1_kernel_matrix(
    X_np: np.ndarray,
    shots: int = 1024,
    exact: bool = True,
    n_workers: int = 1,
    gpu_ids: Optional[List[int]] = None,
) -> np.ndarray:
    X_np = np.asarray(X_np, float)
    N = len(X_np)
    K = np.zeros((N, N), dtype=float)
    if n_workers <= 1:
        with Timer("DQC1 kernel (serial)"):
            for i in range(N):
                K[i, i] = 1.0
                for j in range(i + 1, N):
                    v = dqc1_kernel_value(X_np[i], X_np[j], shots=shots, exact=exact)
                    K[i, j] = K[j, i] = float(np.clip(v, -1.0, 1.0))
        return K
    try:
        from joblib import Parallel, delayed
        pairs = _make_pairs(N)
        def _one(k_idx):
            if gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[k_idx % len(gpu_ids)])
            i, j = pairs[k_idx]
            v = dqc1_kernel_value(X_np[i], X_np[j], shots=shots, exact=exact)
            return (i, j, v)
        with Timer(f"DQC1 kernel (joblib x{n_workers})"):
            results = Parallel(n_jobs=n_workers, backend="loky", prefer="processes")(
                delayed(_one)(k_idx) for k_idx in range(len(pairs))
            )
        for i, j, v in results:
            K[i, j] = K[j, i] = float(np.clip(v, -1.0, 1.0))
        np.fill_diagonal(K, 1.0)
    except Exception as e:
        print(f"[parallel warn] Falling back to serial due to: {e}")
        return build_dqc1_kernel_matrix(X_np, shots=shots, exact=exact, n_workers=1)
    return K

# -------- (unused now, kept for completeness) --------
def _psd_project_and_trace_normalize_t(M_t: t.Tensor, eps: float = 1e-7) -> t.Tensor:
    """
    Robust PSD projection + trace normalization:
      1) Symmetrize
      2) Eigen-decompose, clip negative eigenvalues to 0
      3) Add small diagonal εI
      4) Trace-normalize to get ρ
    """
    M_t = 0.5 * (M_t + M_t.T)
    vals, vecs = t.linalg.eigh(M_t)
    vals = t.clamp(vals, min=0.0)
    M_psd = (vecs @ t.diag(vals) @ vecs.T).real
    M_psd = M_psd + eps * t.eye(M_psd.shape[0], dtype=DTYPE, device=M_psd.device)
    tr = t.trace(M_psd)
    if tr <= 0:
        raise ValueError("PSD projection produced non-positive trace.")
    rho = M_psd / tr
    return rho

# --- FABLE: provide installed package exposing `fable.fable` ---
try:
    from fable import fable as fable_build
except Exception as e:
    raise ImportError("FABLE import failed. Ensure `from fable import fable as fable_build` works.") from e

def fable_block_encode_rho2_t(rho_t: t.Tensor):
    A_t  = rho_t @ rho_t
    A_np = np.asarray(tnp(A_t), dtype=np.float64, order="C")
    out  = fable_build(A_np)
    if isinstance(out, tuple) and len(out) == 2:
        U_A, alpha = out
    else:
        U_A, alpha = out, 1.0
    return U_A, float(alpha)

def _as_controllable_gate(U):
    from qiskit.circuit import Gate
    from qiskit import QuantumCircuit

    if isinstance(U, QuantumCircuit):
        # Convert a QuantumCircuit to a Gate with a sensible label
        return U.to_gate(label=getattr(U, "name", "U_A"))

    if hasattr(U, "definition") and U.definition is not None:
        # Generic Gate-like object with a definition
        g = Gate(getattr(U, "name", "U_A"), U.num_qubits, [])
        g.definition = U.definition
        return g

    if hasattr(U, "to_gate"):
        g = U.to_gate()
        if getattr(g, "definition", None) is None:
            raise TypeError("Block-encoding has no definition; cannot simulate.")
        return g

    raise TypeError(f"Cannot convert {type(U)} to a controllable gate with definition.")


def _run_counts_in_chunks(backend, tqc, total_shots: int, chunk: int = 128):
    remaining = int(total_shots)
    agg = Counter()
    while remaining > 0:
        s = min(chunk, remaining)
        res = backend.run(tqc, shots=s).result()
        cnts = res.get_counts()
        cnts = cnts[0] if isinstance(cnts, list) else cnts
        agg.update(cnts)
        remaining -= s
    return dict(agg)

from qiskit import ClassicalRegister

def _infer_work_and_ancillas(U_A, rho_dim: int, ancilla_qubits=None):
    n_tot = U_A.num_qubits if hasattr(U_A, "num_qubits") else U_A.to_gate().num_qubits
    n_work = int(math.ceil(math.log2(rho_dim)))
    if ancilla_qubits is None:
        m = max(0, n_tot - n_work)
        ancilla_qubits = list(range(m))
    anc_qubits = sorted(set(ancilla_qubits))
    anc_set = set(anc_qubits)
    work_qubits = [q for q in range(n_tot) if q not in anc_set][:n_work]
    if len(work_qubits) != n_work:
        raise ValueError(f"Expected {n_work} work qubits, found {len(work_qubits)}.")
    return work_qubits, anc_qubits, n_work, n_tot

def dqc1_trace_expectation_blockencoded(
    U_A,
    rho_dim: int,
    ancilla_qubits=None,
    shots: int = 128,
    exact: bool = True,
):
    work_qubits, anc_qubits, n_work, n_tot = _infer_work_and_ancillas(U_A, rho_dim, ancilla_qubits)
    c_anc   = ClassicalRegister(len(anc_qubits), "canc")
    c_probe = ClassicalRegister(1, "cprobe")
    qc = QuantumCircuit(1 + n_tot + n_work, c_anc.size + c_probe.size, name="DQC1_purity")
    qc.add_register(c_anc); qc.add_register(c_probe)
    probe = 0; sys0  = 1; env0  = 1 + n_tot
    for k, w in enumerate(work_qubits):
        sys_w = sys0 + w; env_w = env0 + k
        qc.h(sys_w); qc.cx(sys_w, env_w)
    qc.h(probe)
    U_gate = _as_controllable_gate(U_A)
    sys_targets = list(range(sys0, sys0 + n_tot))
    qc.append(U_gate.control(1), [probe] + sys_targets)
    qc.h(probe)
    for j, aq in enumerate(anc_qubits):
        qc.measure(sys0 + aq, j)
    qc.measure(probe, c_anc.size)
    if exact:
        qc_nom = qc.remove_final_measurements(inplace=False)
        psi    = Statevector.from_instruction(qc_nom)
        probs  = psi.probabilities_dict()
        m = len(anc_qubits)
        zero = one = kept0 = kept1 = 0.0
        total_all = 0.0
        for b, p in probs.items():
            s = b.replace(" ", "")
            total_all += p
            probe_bit = s[-1]
            if probe_bit == "0": zero += p
            else: one += p
            if m:
                anc_bits = s[-(1 + m):-1]
                if anc_bits == "0" * m:
                    if probe_bit == "0": kept0 += p
                    else: kept1 += p
        if m and (kept0 + kept1) > 0:
            p0c = kept0 / (kept0 + kept1); p1c = 1.0 - p0c
            expX_cond = p0c - p1c
            p_succ    = (kept0 + kept1) / max(total_all, 1e-12)
            return float(expX_cond), float(p_succ), True
        p0 = zero / max(zero + one, 1e-12); p1 = 1.0 - p0
        return float(p0 - p1), 1.0, False
    backend = make_backend(exact=False, gpu_wanted=True, precision="single")
    tqc = transpile(qc, backend, optimization_level=0)
    cnts = _run_counts_in_chunks(backend, tqc, total_shots=shots, chunk=min(4000, shots))
    total_all = sum(cnts.values())
    m = len(anc_qubits)
    kept0 = kept1 = zero = one = 0
    for bitstr, n in cnts.items():
        s = bitstr.replace(" ", "")
        probe_bit = s[-1]
        if probe_bit == "0": zero += n
        else: one += n
        if m:
            anc = s[-(1 + m):-1]
            if anc == "0" * m:
                if probe_bit == "0": kept0 += n
                else: kept1 += n
    if m and (kept0 + kept1) > 0:
        p0c = kept0 / (kept0 + kept1); p1 = 1.0 - p0c
        return float(p0c - p1), (kept0 + kept1) / max(1, total_all), True
    p0 = zero / max(1, zero + one); p1 = 1.0 - p0
    return float(p0 - p1), 1.0, False

# === spectral compression of cluster kernel before FABLE + DQC1 ===
def renyi2_from_kernel_via_dqc1_purity(
    K_np: np.ndarray,
    purity_shots: int = 128,
    exact_sv_for_purity: bool = True,
    ancilla_qubits: Optional[List[int]] = None,
    max_rank: Optional[int] = None,
) -> Tuple[float, dict]:
    """
    Map the cluster similarity submatrix to a *compressed* density matrix:
      1) Symmetrize K and take eigen-decomposition.
      2) Clip negative eigenvalues.
      3) Keep only the top `max_rank` eigenvalues (if given).
      4) Build a diagonal density matrix ρ_small = diag(λ_i / Σ λ_i).
      5) Call FABLE on ρ_small^2 and estimate Tr(ρ_small^2) via DQC1.
    """
    K_np = np.asarray(K_np, float)
    M_sym = 0.5 * (K_np + K_np.T)
    vals, _ = np.linalg.eigh(M_sym)
    vals = np.clip(vals, 0.0, None)
    if not np.any(vals > 0):
        vals = np.ones(1, dtype=float)
    idx_desc = np.argsort(vals)[::-1]
    vals = vals[idx_desc]
    if max_rank is not None and max_rank > 0 and max_rank < len(vals):
        vals = vals[:max_rank]
    s = float(vals.sum())
    if s <= 0:
        vals = np.ones_like(vals) / float(len(vals))
    else:
        vals = vals / s
    rho_small = np.diag(vals)
    rho_t = npt(rho_small)
    rhoN = int(rho_small.shape[0])
    n_work = int(math.ceil(math.log2(rhoN)))



    if PROFILE_FABLE_PURITY:
        t0 = time.time()
        U_A, alpha = fable_block_encode_rho2_t(rho_t)
        dt = time.time() - t0
        PROFILE_STATS["fable_time"]  += dt
        PROFILE_STATS["fable_calls"] += 1
    else:
        U_A, alpha = fable_block_encode_rho2_t(rho_t)




    _, anc, _, n_tot = _infer_work_and_ancillas(U_A, rhoN, ancilla_qubits)
    anc = anc if anc else None
    MAX_QUBITS_EXACT = 22
    shots_eff = purity_shots
    exact_eff = bool(exact_sv_for_purity)
    if exact_eff and n_tot > MAX_QUBITS_EXACT:
        print(f"[WARN] Purity circuit {n_tot} qubits too large for exact SV; switching to sampled MPS.")
        exact_eff = False
        shots_eff = min(shots_eff, 128)

    if PROFILE_FABLE_PURITY:
        t1 = time.time()
        expX_cond, p_succ, used_cond = dqc1_trace_expectation_blockencoded(
            U_A, rho_dim=rhoN, ancilla_qubits=anc, shots=shots_eff, exact=exact_eff
        )
        dt2 = time.time() - t1
        PROFILE_STATS["purity_time"]  += dt2
        PROFILE_STATS["purity_calls"] += 1
    else:
        expX_cond, p_succ, used_cond = dqc1_trace_expectation_blockencoded(
            U_A, rho_dim=rhoN, ancilla_qubits=anc, shots=shots_eff, exact=exact_eff
        )


    expX_for_trace = expX_cond
    tr_rho2_est = float(alpha) * float(2**n_work) * float(expX_for_trace)
    tr_rho2     = float(np.clip(abs(tr_rho2_est), 1e-12, 1.0))
    H2          = float(-np.log(tr_rho2))
    info = dict(alpha=alpha, n_tot=n_tot, n_work=n_work,
                expX_cond=expX_cond, p_succ=p_succ, used_cond=used_cond,
                Tr_rho2=tr_rho2, comp_rank=len(vals))
    return H2, info

# ============================================================
# -------- DQC1 Entropy Clustering (with reduction) ---------
# ============================================================
class DQC1EntropyCluster:
    """
    Stage 1 (pure quantum, init + ΔH) + Jenssen-style reduction.

    Exposes after .fit(X):
      - self.M_full   : similarity in [0,1] derived from DQC1 kernel
      - self.clusters_: clusters after DQC1-only assignments

    Exposes after .fit_full(X):
      - self.labels_full_ : labels after DQC1 reduction (K*)
      - self.Kstar_       : selected K* (by largest jump in between entropy)
      - self.snapshots_   : clustering snapshots by K
      - self.entropy_log_ : [(K, B_K)] history

    Cluster entropies H2 are evaluated on a *compressed* spectrum of the
    cluster kernel submatrix, with rank <= compress_rank.
    """
    def __init__(self,
                 init_clusters: int = INIT_CLUSTERS,
                 n_init_per_seed: int = N_INIT,
                 kernel_shots: int = DQC1_KERNEL_SHOTS,
                 #exact_sv_kernel: bool = False,
                 exact_sv_kernel=bool(int(os.environ.get("DQC1_EXACT_KERNEL", 1))),
                 purity_shots: int = DQC1_PURITY_SHOTS,
                 #exact_sv_purity: bool = False,
                 exact_sv_purity=bool(int(os.environ.get("DQC1_EXACT_PURITY", 1))),
                 kernel_workers: int = 1,
                 purity_workers: int = 1,
                 gpu_ids: Optional[List[int]] = None,
                 random_state: int = RANDOM_STATE,
                 compress_rank: Optional[int] = DQC1_COMP_RANK):
        self.K0 = int(init_clusters)
        self.N_INIT = int(n_init_per_seed)
        self.kernel_shots = int(kernel_shots)
        self.exact_sv_kernel = bool(exact_sv_kernel)
        self.purity_shots = int(purity_shots)
        self.exact_sv_purity = bool(exact_sv_purity)
        self.kernel_workers = int(kernel_workers)
        self.purity_workers = int(purity_workers)
        self.gpu_ids = list(gpu_ids) if gpu_ids else None
        self.random_state = int(random_state)
        self.rng = np.random.RandomState(self.random_state)
        self.compress_rank = int(compress_rank) if compress_rank is not None else None

    def _compute_full_kernel(self, X_np):
        K = build_dqc1_kernel_matrix(
            X_np,
            shots=self.kernel_shots,
            exact=self.exact_sv_kernel,
            n_workers=self.kernel_workers,
            gpu_ids=self.gpu_ids
        )
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)
        K = np.clip(K, -1.0, 1.0)
        M = 0.5 * (K + 1.0)        
        return M

    def _H2_cluster(self, M_full: np.ndarray, idxs: List[int]) -> float:
        if len(idxs) <= 1:
            return 0.0
        sub = M_full[np.ix_(idxs, idxs)]
        H2, _ = renyi2_from_kernel_via_dqc1_purity(
            sub,
            purity_shots=self.purity_shots,
            exact_sv_for_purity=self.exact_sv_purity,
            max_rank=self.compress_rank
        )
        return H2

    def _delta_H(self, M_full: np.ndarray, base_idxs: List[int], i: int) -> float:
        H_now = self._H2_cluster(M_full, base_idxs) if base_idxs else 0.0
        H_new = self._H2_cluster(M_full, base_idxs + [i])
        return H_new - H_now

    def _delta_H_all_parallel(self, M_full: np.ndarray, clusters: List[List[int]], i_star: int) -> List[float]:
        """
        Parallel ΔH with robust fallback.
        """
        if self.purity_workers <= 1:
            return [self._delta_H(M_full, c, i_star) for c in clusters]

        try:
            from joblib import Parallel, delayed
        except Exception as e:
            print(f"[parallel warn] purity parallel unavailable ({e}); using serial ΔH.")
            return [self._delta_H(M_full, c, i_star) for c in clusters]

        def _one(k_idx: int, base: List[int]):
            if self.gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids[k_idx % len(self.gpu_ids)])
            H_now = self._H2_cluster(M_full, base) if base else 0.0
            H_new = self._H2_cluster(M_full, base + [i_star])
            return H_new - H_now

        try:
            with Timer(f"ΔH batch (joblib x{self.purity_workers})"):
                dHs = Parallel(n_jobs=self.purity_workers, backend="loky", prefer="processes")(
                    delayed(_one)(k, clusters[k]) for k in range(len(clusters))
                )
            return list(dHs)
        except Exception as e:
            print(f"[parallel warn] ΔH parallel failed ({e}); falling back to serial.")
            return [self._delta_H(M_full, c, i_star) for c in clusters]

    def _min_dist_to_cluster_members(self, X_t, unassigned, clusters):
        idt_un = t.as_tensor(unassigned, device=DEVICE, dtype=t.long)
        XU = X_t.index_select(0, idt_un)
        best_i, best_c, best_d2 = None, None, float("inf")
        for c_id, c in enumerate(clusters):
            if not c: continue
            idt_c = t.as_tensor(c, device=DEVICE, dtype=t.long)
            A = X_t.index_select(0, idt_c)
            d2_mat = ((XU.unsqueeze(1) - A.unsqueeze(0)) ** 2).sum(dim=2)
            d2 = d2_mat.min(dim=1).values
            j = int(t.argmin(d2).item())
            cand_d2 = float(d2[j].item())
            if cand_d2 < best_d2:
                best_d2 = cand_d2
                best_i = int(idt_un[j].item())
                best_c = c_id
        return best_i, best_c

    # ---------- between-cluster entropy and reduction ----------
    def _between_entropy(self, M_full: np.ndarray, clusters: List[List[int]]) -> float:
        """
        Jenssen-style between-cluster entropy, but using the DQC1 similarity matrix M_full.
        Constant offsets are omitted (only relative values matter for K*).
        """
        N = M_full.shape[0]
        labels = -np.ones(N, dtype=int)
        for k, c in enumerate(clusters):
            labels[np.asarray(c, dtype=int)] = k
        exist = labels >= 0
        diff  = (labels[:, None] != labels[None, :])
        mask  = exist[:, None] & exist[None, :] & diff
        num   = float((M_full * mask).sum())
        Nk    = np.array([len(c) for c in clusters if len(c) > 0], dtype=float)
        denom = float(2.0 * np.prod(Nk)) if Nk.size > 0 else 1.0
        V     = num / max(denom, 1e-300)
        return -np.log(max(V, 1e-300))

    def _pick_worst_by_between(self, M_full: np.ndarray, clusters: List[List[int]]) -> int:
        """
        Choose the cluster whose removal maximizes between-cluster entropy.
        """
        best_val, best_i = -np.inf, None
        for i in range(len(clusters)):
            rem = [clusters[j] for j in range(len(clusters)) if j != i]
            B = self._between_entropy(M_full, rem)
            if B > best_val:
                best_val, best_i = B, i
        return best_i

    def _reduce(self, X_arr: np.ndarray, clusters: List[List[int]]) -> List[List[int]]:
        """
        Jenssen-style reduction loop:
          - repeatedly remove the 'worst' cluster
          - reassign its samples by ΔH to remaining clusters
          - log between entropy B_K and snapshots for each K
        """
        X = np.asarray(X_arr, float)
        D = X.shape[1]
        self.snapshots_ = {len(clusters): [list(c) for c in clusters]}
        self.entropy_log_ = [(len(clusters), self._between_entropy(self.M_full, clusters))]
        while len(clusters) > 1:
            worst_idx = self._pick_worst_by_between(self.M_full, clusters)
            worst = clusters[worst_idx]
            remaining = [clusters[i] for i in range(len(clusters)) if i != worst_idx]
            if worst:
                means = [X[np.asarray(c)].mean(axis=0) if len(c) else np.zeros(D) for c in remaining]
                means = np.asarray(means)
                W = np.asarray(worst, dtype=int)
                d2 = ((X[W][:, None, :] - means[None, :, :])**2).sum(axis=2)
                order = np.argsort(d2.min(axis=1))
                for loc in order:
                    idx = int(W[loc])
                    best_k, best_dH = None, float("inf")
                    for k, c in enumerate(remaining):
                        dH = self._delta_H(self.M_full, c, idx)
                        if dH < best_dH:
                            best_k, best_dH = k, dH
                    remaining[best_k].append(idx)
            clusters = remaining
            self.entropy_log_.append((len(clusters), self._between_entropy(self.M_full, clusters)))
            self.snapshots_[len(clusters)] = [list(c) for c in clusters]
        return clusters

    # ---------- init + ΔH-only (no reduction) ----------
    def _init_assign(self, X_np: np.ndarray):
        X_t = npt(np.asarray(X_np, float))
        X_t = zero_mean_to_unit_interval_t(X_t)
        self.X_ = tnp(X_t)

        with Timer(f"DQC1 kernel build (map={FEATURE_MAP}, L={LAYERS})"):
            self.M_full = self._compute_full_kernel(self.X_)

        N = self.X_.shape[0]
        rng = self.rng
        seeds = rng.choice(N, self.K0, replace=False)
        clusters = [[int(s)] for s in seeds]
        unassigned = np.setdiff1d(np.arange(N), seeds, assume_unique=True).tolist()

        with Timer("Seed growth"):
            while any(len(c) < self.N_INIT for c in clusters) and unassigned:
                i_star, c_star = self._min_dist_to_cluster_members(npt(self.X_), unassigned, clusters)
                clusters[c_star].append(i_star)
                unassigned.remove(i_star)

        with Timer("ΔH2 assignments (multi-GPU, compressed clusters)"):
            while unassigned:
                i_star, _ = self._min_dist_to_cluster_members(npt(self.X_), unassigned, clusters)
                dHs = self._delta_H_all_parallel(self.M_full, clusters, i_star)
                best_k = int(np.argmin(dHs))
                clusters[best_k].append(i_star)
                unassigned.remove(i_star)

        self.clusters_ = clusters
        return self

    def fit(self, X_np: np.ndarray):
        """
        DQC1 init + ΔH-only (no reduction).
        """
        return self._init_assign(X_np)

    def fit_full(self, X_np: np.ndarray):
        """
        DQC1 FULL pipeline: init + ΔH + reduction + K* selection.
        Reuses any existing init+ΔH state if X_np matches.
        """
        X_np = np.asarray(X_np, float)
        if not hasattr(self, "X_") or self.X_.shape[0] != X_np.shape[0]:
            self._init_assign(X_np)
        clusters_start = [list(c) for c in self.clusters_]
        clusters_final = self._reduce(self.X_, clusters_start)
        # Choose K* via largest jump in between entropy
        self.Kstar_ = first_large_jump_K(self.entropy_log_)
        clusters_star = self.snapshots_.get(self.Kstar_, clusters_final)
        self.labels_full_ = labels_from_clusters(self.X_.shape[0], clusters_star)
        # For completeness, store best between entropy at K*
        for K, B in self.entropy_log_:
            if K == self.Kstar_:
                self.B_best_ = B
                break
        return self

# ============================================================
# -------------- Classical Parzen (FULL PIPELINE) -----------
# ============================================================
class ParzenRenyi2:
    """
    Parzen Rényi-2 pipeline (Jenssen et al.).
    Provides:
      • fit_no_reduce(): init + ΔH only
      • fit_full(): init + ΔH + reduction + K* selection
    """
    def __init__(self,
                 init_clusters: int = INIT_CLUSTERS,
                 min_clusters: int = MIN_CLUSTERS,
                 n_init_per_seed: int = N_INIT,
                 sigma: Optional[float] = None,
                 random_state: int = RANDOM_STATE):
        self.K0  = int(init_clusters)
        self.Kmin = int(min_clusters)
        self.N_INIT = int(n_init_per_seed)

        if sigma is None:
            if len(PARZEN_SIGMA_GRID) > 0:
                self.sigma = float(PARZEN_SIGMA_GRID[0])
            else:
                self.sigma = 0.4
        else:
            self.sigma = float(sigma)

        self.random_state = int(random_state)
        self.rng = np.random.RandomState(self.random_state)

    def _precompute(self, X):
        self.X = zero_mean_to_unit_interval(np.asarray(X, dtype=float))
        self.N, self.D = self.X.shape
        self.D2 = pairwise_distances(self.X, self.X, metric="sqeuclidean")
        self._update_kernel(self.sigma)

    def _update_kernel(self, sigma):
        self.sigma = float(sigma)
        self.M = np.exp(-self.D2 / (4.0 * self.sigma * self.sigma))
        self.CONST = self.D * np.log(2.0 * self.sigma * np.sqrt(np.pi))

    def _H2(self, idxs):
        idxs = np.asarray(idxs, dtype=int)
        if idxs.size == 0:
            return 0.0
        sub = self.M[np.ix_(idxs, idxs)]
        J = float(sub.mean())
        return -np.log(max(J, 1e-300)) + self.CONST

    def _delta_H(self, i, cluster):
        return self._H2(cluster + [i]) - self._H2(cluster)

    def _min_dist_to_cluster_members(self, unassigned, clusters):
        U = np.asarray(unassigned, dtype=int)
        XU = self.X[U]
        best_i, best_c, best_d2 = None, None, np.inf
        for c_id, c in enumerate(clusters):
            A = self.X[np.asarray(c, dtype=int)]
            d2_min = ((XU[:, None, :] - A[None, :, :])**2).sum(axis=2).min(axis=1)
            j = int(np.argmin(d2_min))
            if d2_min[j] < best_d2:
                best_d2 = float(d2_min[j]); best_i = int(U[j]); best_c = c_id
        return best_i, best_c

    def _initial_clustering(self):
        seeds = self.rng.choice(self.N, self.K0, replace=False)
        clusters = [[int(s)] for s in seeds]
        unassigned = np.setdiff1d(np.arange(self.N), seeds, assume_unique=True).tolist()
        while any(len(c) < self.N_INIT for c in clusters) and unassigned:
            i_star, c_star = self._min_dist_to_cluster_members(unassigned, clusters)
            clusters[c_star].append(i_star); unassigned.remove(i_star)
        return clusters, unassigned

    def _label_remaining_by_deltaH(self, clusters, unassigned):
        while unassigned:
            i_star, _ = self._min_dist_to_cluster_members(unassigned, clusters)
            best_k, best_dH = None, float("inf")
            for k, c in enumerate(clusters):
                dH = self._delta_H(i_star, c)
                if dH < best_dH: best_k, best_dH = k, dH
            clusters[best_k].append(i_star); unassigned.remove(i_star)
        return clusters

    def _between_entropy(self, clusters):
        labels = -np.ones(self.N, dtype=int)
        for k, c in enumerate(clusters): labels[np.asarray(c, dtype=int)] = k
        exist = labels >= 0
        diff  = (labels[:, None] != labels[None, :])
        mask  = exist[:, None] & exist[None, :] & diff
        num   = float((self.M * mask).sum())
        Nk    = np.array([len(c) for c in clusters if len(c) > 0], dtype=float)
        denom = float(2.0 * np.prod(Nk)) if Nk.size > 0 else 1.0
        V     = num / max(denom, 1e-300)
        return -np.log(max(V, 1e-300)) + self.CONST

    def _pick_worst_by_between(self, clusters):
        best_val, best_i = -np.inf, None
        for i in range(len(clusters)):
            rem = [clusters[j] for j in range(len(clusters)) if j != i]
            B = self._between_entropy(rem)
            if B > best_val: best_val, best_i = B, i
        return best_i

    def _reduce(self, clusters):
        self.snapshots = {len(clusters): [list(c) for c in clusters]}
        self.entropy_log = [(len(clusters), self._between_entropy(clusters))]
        while len(clusters) > 1:
            worst_idx = self._pick_worst_by_between(clusters)
            worst = clusters[worst_idx]
            remaining = [clusters[i] for i in range(len(clusters)) if i != worst_idx]
            if worst:
                means = [self.X[np.asarray(c)].mean(axis=0) if len(c) else np.zeros(self.D) for c in remaining]
                means = np.asarray(means)
                W = np.asarray(worst, dtype=int)
                d2 = ((self.X[W][:, None, :] - means[None, :, :])**2).sum(axis=2)
                order = np.argsort(d2.min(axis=1))
                for loc in order:
                    idx = int(W[loc])
                    best_k, best_dH = None, float("inf")
                    for k, c in enumerate(remaining):
                        dH = self._delta_H(idx, c)
                        if dH < best_dH:
                            best_k, best_dH = k, dH
                    remaining[best_k].append(idx)
            clusters = remaining
            self.entropy_log.append((len(clusters), self._between_entropy(clusters)))
            self.snapshots[len(clusters)] = [list(c) for c in clusters]
        return clusters

    def _sigma_grid_from_data(self):
        d = np.sqrt(self.D2[np.triu_indices_from(self.D2, 1)])
        qs = np.quantile(d, [0.10, 0.25, 0.40, 0.55, 0.70])
        return np.unique(np.clip(qs/2.0, 1e-6, None))

    def fit_no_reduce(self, X, sigma_grid=None):
        """
        Jenssen-style init + ΔH only (no reduction).
        Uses global defaults when sigma_grid is None.
        """
        self._precompute(X)
        if sigma_grid is None:
            sigma_grid = PARZEN_SIGMA_GRID if len(PARZEN_SIGMA_GRID) > 0 else [self.sigma]
        elif not sigma_grid:
            sigma_grid = [self.sigma]
        best = None
        for s in sigma_grid:
            self._update_kernel(s)
            cl, un = self._initial_clustering()
            cl = self._label_remaining_by_deltaH(cl, un)
            H_sum = sum(self._H2(c) for c in cl)
            if (best is None) or (H_sum < best[0]):
                best = (H_sum, s, [list(c) for c in cl])
        _, self.sigma_, clusters_pre = best
        self.labels_pre_ = labels_from_clusters(self.N, clusters_pre)
        self.clusters_pre_ = clusters_pre
        return self

    def fit_full(self, X, sigma_grid=None, repeats=REPEATS):
        self._precompute(X)
        if sigma_grid is None:
            sigma_grid = self._sigma_grid_from_data()
        elif not sigma_grid:
            sigma_grid = [self.sigma]
        best = None
        for s in sigma_grid:
            self._update_kernel(s)
            for t in range(max(1, repeats)):
                self.rng = np.random.RandomState(self.random_state + t)
                cl, un = self._initial_clustering()
                cl = self._label_remaining_by_deltaH(cl, un)
                cl = self._reduce(cl)
                B = self._between_entropy(cl)
                if (best is None) or (B > best[0]):
                    best = (B, s, [list(c) for c in cl],
                            list(self.entropy_log),
                            {k: [list(cc) for cc in v] for k, v in self.snapshots.items()})
        B_best, sigma_best, clusters_best, entropy_log, snapshots = best
        self._update_kernel(sigma_best)
        Kstar = first_large_jump_K(entropy_log)
        clusters = snapshots.get(Kstar, clusters_best)
        self.labels_ = labels_from_clusters(self.N, clusters)
        self.Kstar_  = Kstar
        self.sigma_  = sigma_best
        self.snapshots_ = snapshots
        self.entropy_log_ = entropy_log
        self.clusters_ = clusters
        self.B_best_ = B_best
        return self

# ---------- Parzen FULL helper ----------
def parzen_full_labels(X, sigma_grid=None, repeats=REPEATS):
    """
    Helper for FULL Parzen pipeline.
    Uses GLOBAL KNOBS.
    """
    if sigma_grid is None:
        sigma_grid = PARZEN_SIGMA_GRID

    prz = ParzenRenyi2(init_clusters=INIT_CLUSTERS,
                       min_clusters=MIN_CLUSTERS,
                       n_init_per_seed=N_INIT,
                       random_state=RANDOM_STATE)

    print("[PARZEN FULL CONFIG]",
          f"K0={prz.K0}, Kmin={prz.Kmin}, N_INIT={prz.N_INIT}, "
          f"sigma_init={prz.sigma}, repeats={repeats}, "
          f"sigma_grid={sigma_grid}")

    prz.fit_full(X, sigma_grid=sigma_grid, repeats=repeats)
    return prz.labels_, prz.Kstar_, prz.sigma_

# -------------------- Synthetic datasets --------------------
def make_spirals(n=300, noise=None, random_state=0):
    if noise is None:
        noise = DATA_NOISE  

    rng = np.random.RandomState(random_state)
    t = np.linspace(0, 2*np.pi, n // 2)
    x1 = np.vstack([t * np.cos(t), t * np.sin(t)]).T + noise * rng.randn(n // 2, 2)
    x2 = np.vstack([t * np.cos(t + np.pi), t * np.sin(t + np.pi)]).T + noise * rng.randn(n // 2, 2)
    X = np.vstack([x1, x2])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y



def toy_sets(n=300, noise=None, random_state=0):
    if noise is None:
        noise = DATA_NOISE  

    # 1) Spirals
    X1, y1 = make_spirals(n=n, noise=noise, random_state=random_state)

    # 2) Rings with factor=0.2
    X2, y2 = make_circles(
        n_samples=n,
        factor=0.2,
        noise=noise,
        random_state=random_state
    )



    # 3)  moons
    X3, y3 = make_moons(
        n_samples=n,
        noise=noise,
        random_state=random_state
    )

    # 4) circles
    X4, y4 = make_circles(
        n_samples=n,
        factor=0.5,
        noise=noise,
        random_state=random_state
    )

    # 5) Anisotropic blobs
    X5, y5 = make_blobs(
        n_samples=n,
        centers=[[0, 0], [3, 0], [0, 3]],
        cluster_std=[0.30, 0.80, 0.40],
        random_state=random_state,
    )
    return (
        (X1, y1, "spirals"),
        (X2, y2, "rings_factor02"),
        (X3, y3, "moons"),
        (X4, y4, "circles"),
        (X5, y5, "anistropic_blobs"),
    )


# -------------------- Benchmark runners (with plot saving) --------------------
from datetime import datetime

def run_synthetic_and_save(tag: str, fmap: str = None, kernel_name: str = None):
    fmap = (os.environ.get("FEATURE_MAP", FEATURE_MAP) if fmap is None else fmap).strip().lower()
    kernel_name = (KERNEL_NAME if kernel_name is None else kernel_name)

    all_sets = toy_sets(n=300, noise=None, random_state=RANDOM_STATE)

    name_map = {nm: (X, y, nm) for (X, y, nm) in all_sets}
    requested = [s.strip() for s in SYN_DATASETS.split("+") if s.strip()]
    sets = []
    for nm in requested:
        if nm in name_map:
            sets.append(name_map[nm])
        else:
            print(f"[WARN] synthetic dataset '{nm}' not found; available: {list(name_map.keys())}")
    if not sets:
        print("[WARN] No valid SYN_DATASETS found; falling back to all synthetic sets.")
        sets = list(all_sets)

    fig, axes = plt.subplots(len(sets), 3, figsize=(16, 5 * len(sets)), constrained_layout=False)
    axes = np.atleast_2d(axes)

    left_label = f"FEATURE MAP: {fmap.upper()} | KERNEL: {kernel_name.upper()}"
    plt.tight_layout(rect=(0.115, 0.02, 0.995, 0.98))
    fig.text(0.09, 0.5, left_label, va="center", ha="left",
             rotation=90, fontsize=12, fontweight="bold")


    for row, (X, y, name) in enumerate(sets):
        dqc = DQC1EntropyCluster(
            init_clusters=INIT_CLUSTERS,
            n_init_per_seed=N_INIT,
            kernel_shots=DQC1_KERNEL_SHOTS,
            exact_sv_kernel=bool(int(os.environ.get("DQC1_EXACT_KERNEL", 1))),
            purity_shots=DQC1_PURITY_SHOTS,
            exact_sv_purity=bool(int(os.environ.get("DQC1_EXACT_PURITY", 1))),
            kernel_workers=DQC1_KERNEL_WORKERS,
            purity_workers=DQC1_PURITY_WORKERS,
            gpu_ids=[int(s) for s in os.environ.get("DQC1_GPU_IDS", "0,1").split(",") if s.strip().isdigit()],
            random_state=RANDOM_STATE,
            compress_rank=DQC1_COMP_RANK,
        )

        print("[DQC1 CONFIG]",
              f"K0={dqc.K0}, N_INIT={dqc.N_INIT}, "
              f"compress_rank={dqc.compress_rank}")

        # DQC1 init + ΔH-only (metrics only)
        dqc.fit(X)
        dqc_clusters_init = dqc.clusters_
        Xn = zero_mean_to_unit_interval(X)
        dqc_labels_init = labels_from_clusters(X.shape[0], dqc_clusters_init)
        s_dqc_init = eval_scores(Xn, y, dqc_labels_init)
        print_scores(f"DQC1 (ΔH-only)-{name}", s_dqc_init)

        # DQC1 FULL (init + ΔH + reduction + K*)
        dqc.fit_full(X)
        dqc_labels_full = dqc.labels_full_
        s_dqc_full = eval_scores(Xn, y, dqc_labels_full)
        print_scores(f"DQC1-FULL-{name} (K*={dqc.Kstar_})", s_dqc_full)

        # Parzen FULL (Gaussian)
        labels_full, Kstar_full, sigma_full = parzen_full_labels(
            X, sigma_grid=PARZEN_SIGMA_GRID, repeats=REPEATS
        )
        s_parzen_full = eval_scores(Xn, y, labels_full)
        print_scores(f"Parzen-FULL-(Gaussian)-{name} (K*={Kstar_full}, σ={sigma_full:.2f})", s_parzen_full)

        # K-Means baseline
        km_K = len(np.unique(y))
        km_labels = KMeans(n_clusters=km_K, n_init=20, random_state=RANDOM_STATE).fit_predict(Xn)
        s_km = eval_scores(Xn, y, km_labels)
        print_scores(f"KMeans-{name}", s_km)

        # Panels (3 columns): DQC1 FULL, Parzen FULL, K-Means
        ax1, ax2, ax3 = axes[row, 0], axes[row, 1], axes[row, 2]

        ax1.scatter(X[:, 0], X[:, 1], c=dqc_labels_full, s=10, cmap="tab10")
        ax1.set_title(f"DQC1 FULL — {name} (K*={dqc.Kstar_})", fontsize=11, pad=6); ax1.grid(True, alpha=.3)
        ax1.set_xlabel("x1"); ax1.set_ylabel("x2")

        ax2.scatter(X[:, 0], X[:, 1], c=labels_full, s=10, cmap="tab10")
        ax2.set_title(f"Parzen FULL (Gauss) — {name} (K*={Kstar_full}, σ={sigma_full:.2f})", fontsize=11, pad=6); ax2.grid(True, alpha=.3)
        ax2.set_xlabel("x1"); ax2.set_ylabel("x2")

        ax3.scatter(X[:, 0], X[:, 1], c=km_labels, s=10, cmap="tab10")
        ax3.set_title(f"K-Means — {name} (K={s_km['K']})", fontsize=11, pad=6); ax3.grid(True, alpha=.3)
        ax3.set_xlabel("x1"); ax3.set_ylabel("x2")

    plt.tight_layout(rect=(0.10, 0.02, 0.995, 0.98))
    out = f"{tag}_synthetics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.show()
    print(f"[saved] {out}")


# ---------------- main ----------------
if __name__ == "__main__":
    fmap = os.environ.get("FEATURE_MAP", FEATURE_MAP).strip().lower()
    kernel_name = KERNEL_NAME
    tag = f"{kernel_name.lower()}_{fmap}_L{LAYERS}"

    print(f"[CONFIG] FEATURE_MAP={fmap}, LAYERS={LAYERS}, "
          f"PARZEN_SIGMA_GRID={PARZEN_SIGMA_GRID}, TAG={tag}, "
          f"KERNEL_NAME={kernel_name}, DQC1_COMP_RANK={DQC1_COMP_RANK}, "
          f"INIT_CLUSTERS={INIT_CLUSTERS}, MIN_CLUSTERS={MIN_CLUSTERS}, "
          f"N_INIT={N_INIT}, REPEATS={REPEATS}, RANDOM_STATE={RANDOM_STATE}")

    print(f"=== Synthetic datasets ({SYN_DATASETS}) ===")
    run_synthetic_and_save(tag)



    if PROFILE_FABLE_PURITY and (PROFILE_STATS["fable_calls"] or PROFILE_STATS["purity_calls"]):
        print("\n=== FABLE + DQC1 Purity Timing Summary ===")
        if PROFILE_STATS["fable_calls"]:
            ft = PROFILE_STATS["fable_time"]
            fc = PROFILE_STATS["fable_calls"]
            print(f"FABLE block-encoding : {fc} calls, total {ft:.3f}s, avg {ft / fc:.4f}s")
        if PROFILE_STATS["purity_calls"]:
            pt = PROFILE_STATS["purity_time"]
            pc = PROFILE_STATS["purity_calls"]
            print(f"DQC1 purity Hadamard : {pc} calls, total {pt:.3f}s, avg {pt / pc:.4f}s")