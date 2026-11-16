"""
verify_mcdm_methods.py - checks the implementations in mcdm_methods.py

1. The heuristic IPS stored in the decision matrices equals
       IPS = C1 * (C2 + C3 + C4) * (0.5 + C5)
2. TOPSIS, VIKOR, PROMETHEE II, WSM, EDAS, MABAC, MARCOS, CRITIC and
   Entropy weights agree with the independent reference library pymcdm
   (Kizielewicz, Shekhovtsov & Salabun, 2023, SoftwareX) on random matrices
   and on the real data. NR-TOPSIS matches its closed form (Li et al., 2023)
   and does not reverse ranks when alternatives are deleted.
3. WS coefficient agrees with pymcdm; Kendall's W equals 1 for identical
   rankings and matches the closed form for two rankings
   (W = (rho_S + 1) / 2 without ties).
4. The PROMETHEE II flow shortcut equals plain pairwise loops for the
   Gaussian and V-shape preference functions, and V-shape with p = range
   reproduces the WSM ranking (reason why Gaussian is the default).

Run:  python analysis/verify_mcdm_methods.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcdm_methods as M  # noqa: E402

from pymcdm import methods as pm, weights as pw  # noqa: E402
from pymcdm.correlations import ws as pm_ws  # noqa: E402
from pymcdm.normalizations import minmax_normalization, vector_normalization  # noqa: E402

TOL = 1e-9
ok = True


def pm_promethee_vshape(X, w, types, p):
    """
    pymcdm reference for PROMETHEE II (V-shape, threshold p). The pymcdm API
    for the thresholds differs between versions (p passed to the call or to
    the constructor); both forms are tried. Returns None if neither exists.
    """
    try:
        return pm.PROMETHEE_II("vshape")(X, w, types, p=p)
    except TypeError:
        pass
    try:
        return pm.PROMETHEE_II("vshape", p=p)(X, w, types)
    except TypeError:
        return None


def check(name, a, b, tol=TOL):
    global ok
    if b is None:
        print("[SKIP] {:<47s} (not available in this pymcdm version; "
              "covered by the pairwise-loop check below)".format(name))
        return
    err = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    passed = err < tol
    ok &= passed
    print("[{}] {:<48s} max abs diff = {:.2e}".format(
        "OK " if passed else "FAIL", name, err))


# ---------------------------------------------------------------- 1. IPS
# uses the decision matrices written by mcdm.py, if present
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
CRIT = ["C1_pagerank", "C2_modal_impact", "C3_poi_loss", "C4_fragmentation",
        "C5_compensation_failure"]
REAL = {}
for city in ("modena", "ferrara", "pesaro"):
    path = os.path.join(OUT, city, "decision_matrix.csv")
    if not os.path.exists(path):
        print("[SKIP] decision matrix of {} not found (run analysis/mcdm.py)".format(city))
        continue
    dm = pd.read_csv(path)
    REAL[city] = dm
    ips = (dm["C1_pagerank"] * (dm["C2_modal_impact"] + dm["C3_poi_loss"]
                                + dm["C4_fragmentation"])
           * (0.5 + dm["C5_compensation_failure"]))
    check("IPS reproduced from criteria ({})".format(city),
          ips, dm["IPS_original"], tol=1e-6)

# ------------------------------------------------- 2. methods vs pymcdm
rng = np.random.default_rng(0)
types = np.ones(5)
for t in range(3):
    X = rng.uniform(0.1, 10, size=(30, 5))
    w = rng.dirichlet(np.ones(5))
    p = X.max(axis=0) - X.min(axis=0)
    check("TOPSIS trial {}".format(t), M.topsis(X, w),
          pm.TOPSIS(vector_normalization)(X, w, types))
    check("VIKOR Q trial {}".format(t), M.vikor(X, w)[2],
          pm.VIKOR()(X, w, types))
    Fv = M.promethee_criterion_flows(X, preference="vshape")
    check("PROMETHEE II V-shape trial {}".format(t), M.promethee_ii(X, w, Fv),
          pm_promethee_vshape(X, w, types, p))
    check("WSM (min-max) trial {}".format(t), M.wsm(X, w),
          pm.WSM(minmax_normalization)(X, w, types))
    check("EDAS trial {}".format(t), M.edas(X, w), pm.EDAS()(X, w, types))
    check("MABAC trial {}".format(t), M.mabac(X, w), pm.MABAC()(X, w, types))
    check("MARCOS trial {}".format(t), M.marcos(X, w), pm.MARCOS()(X, w, types))
    # NR-TOPSIS against its closed form (Li et al., 2023, Eqs. 40-46)
    lo, hi = X.min(axis=0), X.max(axis=0)
    V = (X - lo) / (hi - lo) * w
    dp = np.sqrt(((V - w) ** 2).sum(axis=1))
    dn = np.sqrt((V ** 2).sum(axis=1))
    check("NR-TOPSIS closed form trial {}".format(t),
          M.nr_topsis(X, w, (lo, hi)), dn / (dp + dn))
    check("Contribution factors sum to 1 trial {}".format(t),
          M.contribution_factors(X, w, (lo, hi)).sum(), 1.0)
    check("CRITIC weights trial {}".format(t), M.critic_weights(X),
          pw.critic_weights(X))
    # pymcdm sets entropy to 0 when a column contains a zero; no zeros here
    check("Entropy weights trial {}".format(t), M.entropy_weights(X),
          pw.entropy_weights(X))
    x = rng.permutation(30) + 1.0
    y = rng.permutation(30) + 1.0
    check("WS coefficient trial {}".format(t), M.ws_coefficient(x, y),
          pm_ws(x, y))
    check("Kendall W, two rankings = (rho+1)/2 trial {}".format(t),
          M.kendall_w([x, y]), (spearmanr(x, y)[0] + 1) / 2)

check("Kendall W of identical rankings = 1",
      M.kendall_w([np.arange(1, 11)] * 3), 1.0)

# ------------------ 4. PROMETHEE: flows shortcut vs plain pairwise loops
X = rng.uniform(0, 1, size=(25, 5))
w = rng.dirichlet(np.ones(5))
m = X.shape[0]
s = X.std(axis=0, ddof=1)
p = X.max(axis=0) - X.min(axis=0)


def pref_gauss(d, sj):
    return 1 - np.exp(-d ** 2 / (2 * sj ** 2)) if d > 0 else 0.0


def pref_v(d, pj):
    return min(max(d / pj, 0.0), 1.0)


for name, pf, par in (("gaussian", pref_gauss, s), ("vshape", pref_v, p)):
    phi = np.zeros(m)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            pab = sum(w[j] * pf(X[a, j] - X[b, j], par[j]) for j in range(5))
            pba = sum(w[j] * pf(X[b, j] - X[a, j], par[j]) for j in range(5))
            phi[a] += (pab - pba) / (m - 1)
    F = M.promethee_criterion_flows(X, preference=name)
    check("PROMETHEE {} shortcut = pairwise loops".format(name),
          M.promethee_ii(X, w, F), phi)

# V-shape with p = range reproduces the WSM ranking (documented property)
Fv = M.promethee_criterion_flows(X, preference="vshape")
check("PROMETHEE V-shape(p=range) ranks = WSM ranks",
      M.rank_from_score(M.promethee_ii(X, w, Fv)),
      M.rank_from_score(M.wsm(X, w)))

# --------------- NR-TOPSIS: deleting alternatives cannot reverse ranks
Xr = rng.uniform(0, 1, size=(60, 5))
wr = rng.dirichlet(np.ones(5))
b = (np.zeros(5), np.ones(5))
full = M.rank_from_score(M.nr_topsis(Xr, wr, b))
keep = np.sort(rng.choice(60, 40, replace=False))
sub = M.rank_from_score(M.nr_topsis(Xr[keep], wr, b))
check("NR-TOPSIS order unchanged after deleting 20 alternatives",
      M.rankdata(full[keep]), sub)

# -------------------------------------------- methods on the real data
if "pesaro" in REAL:
    X = REAL["pesaro"][CRIT].values
    w = M.critic_weights(X)
    check("TOPSIS on Pesaro data", M.topsis(X, w),
          pm.TOPSIS(vector_normalization)(X, w, types))
    check("VIKOR on Pesaro data", M.vikor(X, w)[2], pm.VIKOR()(X, w, types))
    check("PROMETHEE II (V-shape) on Pesaro data",
          M.promethee_ii(X, w, M.promethee_criterion_flows(X, preference="vshape")),
          pm_promethee_vshape(X, w, types, X.max(axis=0) - X.min(axis=0)))
    check("WSM on Pesaro data", M.wsm(X, w),
          pm.WSM(minmax_normalization)(X, w, types))
    check("EDAS on Pesaro data", M.edas(X, w), pm.EDAS()(X, w, types))
    # pymcdm computes the MABAC geometric mean as prod(.)^(1/m), which underflows
    # to 0 for hundreds of alternatives; mcdm_methods uses the log-space form.
    # Compared on the first 40 roads (no underflow), and the log form is checked
    # against the definition on the full matrix.
    check("MABAC on Pesaro data (40 roads)", M.mabac(X[:40], w),
          pm.MABAC()(X[:40], w, types))
    Vm = (M.minmax_normalize(X) + 1.0) * w
    Gm = np.array([np.exp(np.mean(np.log(Vm[:, j]))) for j in range(Vm.shape[1])])
    check("MABAC border area = geometric mean (full data)",
          M.mabac(X, w), (Vm - Gm).sum(axis=1))
    check("MARCOS on Pesaro data", M.marcos(X, w), pm.MARCOS()(X, w, types))
    check("CRITIC on Pesaro data", w, pw.critic_weights(X))

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
