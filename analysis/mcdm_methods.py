"""
mcdm_methods.py - weighting, aggregation and sensitivity methods of the MCDM
model, used by mcdm.py.

    weights          CRITIC, Entropy, CRITIC-Entropy, Equal
    core methods     NR-TOPSIS, VIKOR, PROMETHEE II -> consensus ranking
    comparison       TOPSIS, WSM (SAW), EDAS, MABAC, MARCOS
    analyses         rank similarity (Spearman, Kendall, WS coefficient,
                     Kendall's W), weight scenarios, one-at-a-time weight
                     changes, criteria groups, rank reversal, method
                     parameters, SMAA-2, criteria contribution, benchmark
                     comparison

Method references:
    NR-TOPSIS     Yang (2020); as applied in Li et al. (2023, ESWA)
    TOPSIS        Hwang & Yoon (1981), vector normalisation
    VIKOR         Opricovic & Tzeng (2004)
    PROMETHEE II  Brans & Vincke (1985), Gaussian preference function
    WSM / SAW     Fishburn (1967)
    EDAS          Keshavarz Ghorabaee et al. (2015)
    MABAC         Pamucar & Cirovic (2015)
    MARCOS        Stevic et al. (2020)
    CRITIC        Diakoulaki, Mavrotas & Papayannakis (1995)
    WS coeff.     Salabun & Urbaniak (2020)
    SMAA-2        Lahdelma & Salminen (2001)
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, kendalltau, pearsonr, rankdata, spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Main runs (physical-length routing cost, 25% closure) = paper results
# Stress-weighted routing runs
# Closure-depth runs (available for Modena)


CRITERIA = ["C1_pagerank", "C2_modal_impact", "C3_poi_loss",
            "C4_fragmentation", "C5_compensation_failure"]
# Criteria groups (for criteria-group scenarios)
CRITERIA_GROUPS = {
    "network": ["C1_pagerank"],
    "disruption_impact": ["C2_modal_impact", "C3_poi_loss", "C4_fragmentation"],
    "cycling_compensation": ["C5_compensation_failure"],
}
# Domain bounds [m_j, M_j] for NR-TOPSIS (Yang, 2020). C2, C3, C5 are shares
# in [0, 1] by definition; C4 is a sum of three terms capped at 0.2 each;
# C1 (PageRank) has no theoretical bound, so its observed range is used.
DOMAIN_BOUNDS = {
    "C2_modal_impact": (0.0, 1.0),
    "C3_poi_loss": (0.0, 1.0),
    "C4_fragmentation": (0.0, 0.6),
    "C5_compensation_failure": (0.0, 1.0),
}
# Car-only criteria: what a car-based assessment can see
CAR_CRITERIA = ["C1_pagerank", "Ccar_impact", "Ccar_poi_loss",
                "Ccar_fragmentation"]


CORE_METHODS = ["NR_TOPSIS", "VIKOR", "PROMETHEE_II"]
EXTRA_METHODS = ["TOPSIS", "WSM", "EDAS", "MABAC", "MARCOS"]
ALL_METHODS = CORE_METHODS + EXTRA_METHODS
WEIGHT_SCHEMES = ["critic", "entropy", "critic_entropy", "equal"]
TOP_K = (10, 20)
PROMETHEE_PREFERENCE = "gaussian"


# ---------------------------------------------------------------------------
# 1. Data and decision matrix
# ---------------------------------------------------------------------------


def domain_bounds(X, criteria=CRITERIA):
    """[m_j, M_j] for NR-TOPSIS; observed range where no bound is defined."""
    lo, hi = X.min(axis=0).astype(float), X.max(axis=0).astype(float)
    for j, c in enumerate(criteria):
        if c in DOMAIN_BOUNDS:
            lo[j], hi[j] = DOMAIN_BOUNDS[c]
    return lo, hi


# ---------------------------------------------------------------------------
# 2. Weighting methods
# ---------------------------------------------------------------------------

def minmax_normalize(X):
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (X - lo) / rng


def critic_weights(X):
    """CRITIC (Diakoulaki et al., 1995): C_j = sigma_j sum_k (1 - r_jk)."""
    N = minmax_normalize(X)
    sigma = N.std(axis=0, ddof=1)
    r = np.nan_to_num(np.corrcoef(N, rowvar=False), nan=0.0)
    C = sigma * np.sum(1.0 - r, axis=0)
    return C / C.sum()


def entropy_weights(X):
    """Shannon entropy weights, 0 ln 0 := 0."""
    X = np.asarray(X, dtype=float)
    m = X.shape[0]
    colsum = np.where(X.sum(axis=0) == 0, 1.0, X.sum(axis=0))
    P = X / colsum
    with np.errstate(divide="ignore", invalid="ignore"):
        PlnP = np.where(P > 0, P * np.log(P), 0.0)
    E = -PlnP.sum(axis=0) / np.log(m)
    d = 1.0 - E
    return d / d.sum()


def critic_entropy_weights(X):
    """Combined objective weights, multiplicative synthesis."""
    w = critic_weights(X) * entropy_weights(X)
    return w / w.sum()


def equal_weights(X):
    n = np.asarray(X).shape[1]
    return np.full(n, 1.0 / n)


WEIGHT_FUNCS = {"critic": critic_weights, "entropy": entropy_weights,
                "critic_entropy": critic_entropy_weights,
                "equal": equal_weights}


# ---------------------------------------------------------------------------
# 3. MCDM methods (all criteria are benefit criteria)
# ---------------------------------------------------------------------------

def nr_topsis(X, w, bounds):
    """
    NR-TOPSIS (Yang, 2020; Li et al., 2023, Eqs. 40-46).
        a_ij = (x_ij - m_j) / (M_j - m_j)    fixed domain bounds [m_j, M_j]
        v_ij = w_j a_ij,  v+_j = w_j,  v-_j = 0
        S_i  = d-_i / (d+_i + d-_i)                       higher = better
    The normalisation does not depend on the other alternatives, so adding
    or deleting alternatives cannot change the order of the others.
    """
    lo, hi = bounds
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    A = np.clip((np.asarray(X, float) - lo) / rng, 0.0, 1.0)
    V = A * w
    d_pos = np.sqrt(((V - w) ** 2).sum(axis=1))
    d_neg = np.sqrt((V ** 2).sum(axis=1))
    den = d_pos + d_neg
    return d_neg / np.where(den == 0, 1.0, den)


def topsis(X, w):
    """Classic TOPSIS (Hwang & Yoon, 1981), vector normalisation."""
    X = np.asarray(X, dtype=float)
    norm = np.sqrt((X ** 2).sum(axis=0))
    V = (X / np.where(norm == 0, 1.0, norm)) * w
    d_pos = np.sqrt(((V - V.max(axis=0)) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - V.min(axis=0)) ** 2).sum(axis=1))
    den = d_pos + d_neg
    return d_neg / np.where(den == 0, 1.0, den)


def vikor(X, w, v=0.5):
    """VIKOR (Opricovic & Tzeng, 2004). Returns S, R, Q (lower = better)."""
    X = np.asarray(X, dtype=float)
    f_best, f_worst = X.max(axis=0), X.min(axis=0)
    rng = np.where(f_best - f_worst == 0, 1.0, f_best - f_worst)
    D = w * (f_best - X) / rng
    S, R = D.sum(axis=1), D.max(axis=1)
    s_rng = (S.max() - S.min()) or 1.0
    r_rng = (R.max() - R.min()) or 1.0
    Q = v * (S - S.min()) / s_rng + (1 - v) * (R - R.min()) / r_rng
    return S, R, Q


def vikor_conditions(S, R, Q):
    """VIKOR acceptable advantage (C1) and acceptable stability (C2)."""
    m = len(Q)
    order = np.argsort(Q, kind="mergesort")
    dq = 1.0 / (m - 1)
    best, second = order[0], order[1]
    c1 = bool(Q[second] - Q[best] >= dq)
    c2 = bool(best == np.argmin(S) or best == np.argmin(R))
    if c1:
        comp = [int(best)] if c2 else [int(best), int(second)]
    else:
        comp = [int(i) for i in order if Q[i] - Q[best] < dq]
    return {"DQ": dq, "C1_acceptable_advantage": c1,
            "C2_acceptable_stability": c2, "compromise_set_idx": comp}


def promethee_criterion_flows(X, preference=None, s_scale=1.0):
    """
    Single-criterion net flows of PROMETHEE II (Brans & Vincke, 1985).
      Gaussian (type VI): P = 1 - exp(-d^2 / (2 s_j^2)), s_j = s_scale * std_j
      V-shape (type III): P = min(d / p_j, 1), p_j = range of criterion j
    phi(a) = sum_j w_j phi_j(a) is linear in the weights, so the flows are
    computed once per decision matrix. With V-shape and p_j = range,
    PROMETHEE II reproduces the WSM ranking exactly (checked in
    verify_mcdm_methods.py); the Gaussian function is therefore the default.
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    preference = preference or PROMETHEE_PREFERENCE
    F = np.empty((m, n))
    if preference == "vshape":
        p = X.max(axis=0) - X.min(axis=0)
        p = np.where(p == 0, 1.0, p)
    else:
        s = s_scale * X.std(axis=0, ddof=1)
        s = np.where(s == 0, 1.0, s)
    for j in range(n):
        d = X[:, j][:, None] - X[:, j][None, :]
        if preference == "vshape":
            P = np.clip(d / p[j], 0.0, 1.0)
        else:
            P = np.where(d > 0, 1.0 - np.exp(-(d ** 2) / (2.0 * s[j] ** 2)), 0.0)
        F[:, j] = (P.sum(axis=1) - P.sum(axis=0)) / (m - 1)
    return F


def promethee_ii(X, w, flows=None):
    F = promethee_criterion_flows(X) if flows is None else flows
    return F @ np.asarray(w, dtype=float)


def wsm(X, w):
    """WSM / SAW, min-max normalisation."""
    return minmax_normalize(X) @ np.asarray(w, dtype=float)


def edas(X, w):
    """EDAS (Keshavarz Ghorabaee et al., 2015)."""
    X = np.asarray(X, dtype=float)
    av = np.where(X.mean(axis=0) == 0, 1.0, X.mean(axis=0))
    pda = np.maximum(X - av, 0) / av
    nda = np.maximum(av - X, 0) / av
    sp, sn = pda @ w, nda @ w
    return (sp / (sp.max() or 1.0) + 1.0 - sn / (sn.max() or 1.0)) / 2.0


def mabac(X, w):
    """
    MABAC (Pamucar & Cirovic, 2015).
        v_ij = w_j (n_ij + 1),  g_j = (prod_i v_ij)^(1/m),  S_i = sum_j (v_ij - g_j)
    The geometric mean is computed in log space (same value, no underflow).
    """
    V = (minmax_normalize(X) + 1.0) * w
    G = np.exp(np.log(V).mean(axis=0))
    return (V - G).sum(axis=1)


def marcos(X, w):
    """
    MARCOS (Stevic et al., 2020), benefit criteria.
    Ideal AI = column max, anti-ideal AAI = column min; n_ij = x_ij / x_AI.
        K-_i = S_i / S_AAI,  K+_i = S_i / S_AI
        f(K_i) = (K+ + K-) / (1 + (1 - f(K+))/f(K+) + (1 - f(K-))/f(K-))
    """
    X = np.asarray(X, dtype=float)
    ai, aai = X.max(axis=0), X.min(axis=0)
    ai = np.where(ai == 0, 1.0, ai)
    S = (X / ai) @ w
    S_ai, S_aai = (ai / ai) @ w, (aai / ai) @ w
    k_neg, k_pos = S / S_aai, S / S_ai
    f_pos = k_neg / (k_pos + k_neg)
    f_neg = k_pos / (k_pos + k_neg)
    return (k_pos + k_neg) / (1 + (1 - f_pos) / f_pos + (1 - f_neg) / f_neg)


HIGHER_IS_BETTER = {"NR_TOPSIS": True, "TOPSIS": True, "VIKOR": False,
                    "PROMETHEE_II": True, "WSM": True, "EDAS": True,
                    "MABAC": True, "MARCOS": True}


def method_score(name, X, w, flows=None, bounds=None, v=0.5):
    if name == "NR_TOPSIS":
        return nr_topsis(X, w, bounds if bounds is not None else domain_bounds(X))
    if name == "TOPSIS":
        return topsis(X, w)
    if name == "VIKOR":
        return vikor(X, w, v=v)[2]
    if name == "PROMETHEE_II":
        return promethee_ii(X, w, flows)
    if name == "WSM":
        return wsm(X, w)
    if name == "EDAS":
        return edas(X, w)
    if name == "MABAC":
        return mabac(X, w)
    if name == "MARCOS":
        return marcos(X, w)
    raise ValueError(name)


def rank_from_score(score, higher_is_better=True):
    """Rank 1 = best; ties receive the average rank."""
    s = np.asarray(score, dtype=float)
    return rankdata(-s if higher_is_better else s, method="average")


def ordinal_rank(score, higher_is_better=True):
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s if higher_is_better else s, kind="mergesort")
    r = np.empty(len(s), dtype=int)
    r[order] = np.arange(1, len(s) + 1)
    return r


def run_methods(X, w, methods=ALL_METHODS, flows=None, bounds=None, v=0.5):
    if flows is None and "PROMETHEE_II" in methods:
        flows = promethee_criterion_flows(X)
    res = {}
    for mth in methods:
        sc = method_score(mth, X, w, flows, bounds, v)
        res[mth] = (sc, rank_from_score(sc, HIGHER_IS_BETTER[mth]))
    return res


def consensus_rank(rank_list, tiebreak=None):
    """
    Borda-type consensus: order by the mean rank across the core methods;
    equal mean ranks are broken by `tiebreak` (NR-TOPSIS score).
    """
    mean_r = np.mean(np.vstack(rank_list), axis=0)
    if tiebreak is None:
        return mean_r, rankdata(mean_r, method="average")
    order = np.lexsort((-np.asarray(tiebreak, dtype=float), mean_r))
    r = np.empty(len(mean_r))
    r[order] = np.arange(1, len(mean_r) + 1)
    return mean_r, r


def core_consensus(X, w, flows=None, bounds=None):
    res = run_methods(X, w, CORE_METHODS, flows, bounds)
    _, cons = consensus_rank([res[m_][1] for m_ in CORE_METHODS],
                             tiebreak=res["NR_TOPSIS"][0])
    return res, cons


# ---------------------------------------------------------------------------
# 4. Ranking comparison measures
# ---------------------------------------------------------------------------

def ws_coefficient(x, y):
    """WS coefficient (Salabun & Urbaniak, 2020); x = reference ranking."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    N = len(x)
    den = np.maximum(np.abs(x - 1), np.abs(x - N))
    den = np.where(den == 0, 1.0, den)
    return 1.0 - np.sum(2.0 ** (-x) * np.abs(x - y) / den)


def top_set(rank, k):
    return set(np.argsort(rank, kind="mergesort")[:k])


def topk_overlap(x, y, k):
    return len(top_set(x, k) & top_set(y, k)) / float(k)


def kendall_w(rank_matrix):
    """Kendall's coefficient of concordance W, tie-corrected."""
    R = np.asarray(rank_matrix, dtype=float)
    k, n = R.shape
    Rj = R.sum(axis=0)
    S = np.sum((Rj - Rj.mean()) ** 2)
    T = 0.0
    for row in R:
        _, counts = np.unique(row, return_counts=True)
        T += np.sum(counts ** 3 - counts)
    return 12.0 * S / (k ** 2 * (n ** 3 - n) - k * T)


def compare_rankings(ranks, scores=None):
    """Pairwise similarity; Pearson of oriented scores when given."""
    rows = []
    names = list(ranks.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = ranks[a], ranks[b]
            row = {"ranking_A": a, "ranking_B": b,
                   "spearman_rho": spearmanr(x, y)[0],
                   "kendall_tau": kendalltau(x, y)[0],
                   "WS_A_ref": ws_coefficient(x, y)}
            if scores is not None and a in scores and b in scores:
                row["pearson_scores"] = pearsonr(scores[a], scores[b])[0]
            for k in TOP_K:
                row["top{}_overlap".format(k)] = topk_overlap(x, y, k)
            rows.append(row)
    return pd.DataFrame(rows)


def discrimination(score):
    s = np.round(np.asarray(score, dtype=float), 12)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    return {"distinct_scores": int(len(counts)),
            "tied_share": float((counts[inv] > 1).mean()),
            "largest_tie_group": int(counts.max())}


def topk_profile(rank, dm, k):
    sub = dm.iloc[list(top_set(rank, k))]
    return {"k": k,
            "bike_relevant_share": float(sub["bike_relevant"].mean()),
            "bike_only_vulnerable_n": int(sub["bike_only_vulnerable"].sum()),
            "mean_C2_modal_impact": float(sub["C2_modal_impact"].mean()),
            "mean_C3_poi_loss": float(sub["C3_poi_loss"].mean()),
            "mean_C4_fragmentation": float(sub["C4_fragmentation"].mean()),
            "mean_C5_compensation_failure":
                float(sub["C5_compensation_failure"].mean())}


# ---------------------------------------------------------------------------
# 5. Criteria contribution (Chen, 2021)
# ---------------------------------------------------------------------------

def contribution_factors(X, w, bounds):
    """
    Contribution factor of each criterion (CFA) to the NR-TOPSIS distances
    (Chen, 2021, Eq. 7):
        C_j = 0.5 [ sum_i w_j^2 (a_ij - a+_j)^2 / sum_j sum_i w_j^2 (a_ij - a+_j)^2
                  + sum_i w_j^2 (a_ij - a-_j)^2 / sum_j sum_i w_j^2 (a_ij - a-_j)^2 ]
    with a+_j = 1 and a-_j = 0 for the NR-TOPSIS normalisation.
    """
    lo, hi = bounds
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    A = np.clip((np.asarray(X, float) - lo) / rng, 0.0, 1.0)
    pos = (w ** 2) * ((A - 1.0) ** 2).sum(axis=0)
    neg = (w ** 2) * (A ** 2).sum(axis=0)
    return 0.5 * (pos / pos.sum() + neg / neg.sum())


def comprehensiveness(X, w, bounds, scores):
    """CFA of each criterion + Spearman between each criterion and each score."""
    cfa = contribution_factors(X, w, bounds)
    rows = []
    for j, c in enumerate(CRITERIA):
        row = {"criterion": c, "weight": w[j], "CFA_NR_TOPSIS": cfa[j]}
        for name, sc in scores.items():
            row["spearman_with_{}".format(name)] = spearmanr(X[:, j], sc)[0]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Sensitivity analyses
# ---------------------------------------------------------------------------

def perturb_weights(w, j, delta):
    """w_j' = w_j (1 + delta); other weights rescaled proportionally."""
    w = np.asarray(w, dtype=float)
    wj = min(max(w[j] * (1.0 + delta), 0.0), 1.0)
    rest = 1.0 - w[j]
    w_new = w * ((1.0 - wj) / rest) if rest > 0 else w.copy()
    w_new[j] = wj
    return w_new / w_new.sum()


def _similarity_row(base_r, new_r):
    return {"spearman_vs_base": spearmanr(base_r, new_r)[0],
            "WS_vs_base": ws_coefficient(base_r, new_r),
            "top10_retained": topk_overlap(base_r, new_r, 10),
            "top1_unchanged": bool(np.argmin(base_r) == np.argmin(new_r))}


def weight_scenarios(X, w, base_ranks, base_cons, flows, bounds,
                     reductions=tuple(np.round(np.arange(0.05, 1.0, 0.10), 2)),
                     n_top=3):
    """
    Bouraima et al. (2023), Eq. (19): the weight of each of the n_top most
    important criteria is reduced by 5%, 15%, ..., 95%; the other weights
    are increased proportionally so that the sum stays 1
    (w_nb = (1 - w_na) w_b / (1 - w_n)). 3 x 10 = 30 scenarios S1..S30.
    """
    order = np.argsort(-w)[:n_top]
    rows, ranks_store, sid = [], [], 0
    for j in order:
        for red in reductions:
            sid += 1
            w_new = perturb_weights(w, j, -red)
            res, cons = core_consensus(X, w_new, flows, bounds)
            ranks_store.append(("S{}".format(sid), cons))
            for mth in CORE_METHODS + ["Consensus"]:
                r_new = cons if mth == "Consensus" else res[mth][1]
                r_base = base_cons if mth == "Consensus" else base_ranks[mth]
                row = {"scenario": "S{}".format(sid), "criterion": CRITERIA[j],
                       "reduction": red, "method": mth}
                row.update({c: w_new[k] for k, c in enumerate(CRITERIA)})
                row.update(_similarity_row(r_base, r_new))
                rows.append(row)
    return pd.DataFrame(rows), ranks_store


def sensitivity_oat(X, w, base_ranks, flows, bounds,
                    deltas=(-0.5, -0.3, -0.1, 0.1, 0.3, 0.5)):
    rows = []
    for j, c in enumerate(CRITERIA):
        for dlt in deltas:
            w_new = perturb_weights(w, j, dlt)
            res = run_methods(X, w_new, CORE_METHODS, flows, bounds)
            for mth in CORE_METHODS:
                row = {"criterion": c, "delta": dlt, "weight_new": w_new[j],
                       "method": mth}
                row.update(_similarity_row(base_ranks[mth], res[mth][1]))
                rows.append(row)
    return pd.DataFrame(rows)


def criteria_group_scenarios(X, w, base_cons, bounds):
    """
    Li et al. (2023), Sec. 5.1.3: rankings computed on criteria subsets with
    the ratios of the retained weights kept constant (renormalised to 1).
    Leave-one-criterion-out (5 scenarios) and group-only subsets.
    """
    subsets = {"without_{}".format(c): [k for k in CRITERIA if k != c]
               for c in CRITERIA}
    subsets["disruption_impact_only"] = CRITERIA_GROUPS["disruption_impact"]
    subsets["network_and_impact"] = (CRITERIA_GROUPS["network"]
                                     + CRITERIA_GROUPS["disruption_impact"])
    subsets["impact_and_cycling"] = (CRITERIA_GROUPS["disruption_impact"]
                                     + CRITERIA_GROUPS["cycling_compensation"])
    rows = []
    lo, hi = bounds
    for name, crit in subsets.items():
        idx = [CRITERIA.index(c) for c in crit]
        ws = w[idx] / w[idx].sum()
        Xs = X[:, idx]
        _, cons = core_consensus(Xs, ws, None, (lo[idx], hi[idx]))
        row = {"scenario": name, "criteria": "+".join(c.split("_")[0] for c in crit)}
        row.update(_similarity_row(base_cons, cons))
        rows.append(row)
    return pd.DataFrame(rows)


def vikor_v_scenarios(X, w, base_rank, vs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    rows = []
    for v in vs:
        r = rank_from_score(vikor(X, w, v=v)[2], False)
        row = {"parameter": "VIKOR_v", "value": v}
        row.update(_similarity_row(base_rank, r))
        rows.append(row)
    return rows


def promethee_scenarios(X, w, base_rank):
    rows = []
    for pref, sc in (("gaussian", 0.5), ("gaussian", 1.0), ("gaussian", 2.0),
                     ("vshape", 1.0)):
        F = promethee_criterion_flows(X, preference=pref, s_scale=sc)
        r = rank_from_score(promethee_ii(X, w, F), True)
        row = {"parameter": "PROMETHEE_preference",
               "value": "{}{}".format(pref, "" if pref == "vshape"
                                      else " s={}*std".format(sc))}
        row.update(_similarity_row(base_rank, r))
        rows.append(row)
    return rows


def rank_reversal_worst_deletion(X, w, bounds, max_share=0.5,
                                 methods=ALL_METHODS):
    """
    Li et al. (2023); Bouraima et al. (2023): the worst-ranked alternative
    is deleted and the problem is solved again, repeatedly. Weights and
    NR-TOPSIS domain bounds are kept fixed so that only the change of the
    alternative set can alter the order. For each step, the order of the
    remaining alternatives is compared with their order in the full ranking.
    """
    m = X.shape[0]
    n_steps = int(max_share * m)
    rows = []
    for mth in methods:
        full = run_methods(X, w, [mth], bounds=bounds)[mth][1]
        keep = np.arange(m)
        changed_steps, top10_changed = 0, 0
        taus = []
        cur = full.copy()
        for step in range(1, n_steps + 1):
            worst = keep[np.argmax(cur)]
            keep = keep[keep != worst]
            new = run_methods(X[keep], w, [mth], bounds=bounds)[mth][1]
            ref = rankdata(full[keep], method="average")
            tau = kendalltau(ref, new)[0]
            taus.append(tau)
            if tau < 1.0 - 1e-12:
                changed_steps += 1
            if topk_overlap(ref, new, 10) < 1.0 or \
                    list(np.argsort(ref, kind="mergesort")[:10]) != \
                    list(np.argsort(new, kind="mergesort")[:10]):
                top10_changed += 1
            cur = new
        rows.append({"method": mth, "steps": n_steps,
                     "steps_with_any_reversal": changed_steps,
                     "steps_with_top10_order_change": top10_changed,
                     "kendall_tau_min": float(np.min(taus)),
                     "kendall_tau_mean": float(np.mean(taus))})
    return pd.DataFrame(rows)


def rank_reversal_random(X, w, bounds, n_iter=200, frac=0.10, seed=42,
                         methods=ALL_METHODS):
    """Random deletion of 10% of the alternatives (fixed weights/bounds)."""
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    full = run_methods(X, w, methods, bounds=bounds)
    rows = []
    for it in range(n_iter):
        keep = np.sort(rng.choice(m, size=m - int(round(frac * m)), replace=False))
        res = run_methods(X[keep], w, methods, bounds=bounds)
        for mth in methods:
            ref = rankdata(full[mth][1][keep], method="average")
            new = res[mth][1]
            rows.append({"iteration": it, "method": mth,
                         "kendall_tau": kendalltau(ref, new)[0],
                         "top1_unchanged": bool(np.argmin(ref) == np.argmin(new)),
                         "top10_retained": topk_overlap(ref, new, 10)})
    d = pd.DataFrame(rows)
    s = (d.groupby("method")
         .agg(kendall_tau_mean=("kendall_tau", "mean"),
              kendall_tau_min=("kendall_tau", "min"),
              top1_unchanged_share=("top1_unchanged", "mean"),
              top10_retained_mean=("top10_retained", "mean"))
         .reset_index())
    s["n_iter"], s["removed_share"] = n_iter, frac
    return s


def smaa2(X, road_names, flows, bounds, base_ranks, n_samples=2000, seed=42,
          n_rank=10, methods=CORE_METHODS):
    """
    SMAA-2 (Lahdelma & Salminen, 2001) with uniform weights on the simplex:
    rank acceptability b_i^r, holistic acceptability a_i^h (alpha_r =
    (m - r)/(m - 1)), central weight vectors w_i^c.
    """
    rng = np.random.default_rng(seed)
    m, n = X.shape
    W = rng.dirichlet(np.ones(n), size=n_samples)
    alpha = (m - np.arange(1, m + 1)) / (m - 1.0)
    out, stats = {}, []
    for mth in methods:
        b = np.zeros((m, n_rank))
        a_h = np.zeros(m)
        cw_sum = np.zeros((m, n))
        first = np.zeros(m)
        rhos = np.empty(n_samples)
        for s in range(n_samples):
            sc = method_score(mth, X, W[s], flows, bounds)
            r = ordinal_rank(sc, HIGHER_IS_BETTER[mth])
            a_h += alpha[r - 1]
            top = r <= n_rank
            b[np.where(top)[0], r[top] - 1] += 1
            i1 = np.where(r == 1)[0][0]
            first[i1] += 1
            cw_sum[i1] += W[s]
            rhos[s] = spearmanr(base_ranks[mth], r)[0]
        b /= n_samples
        a_h /= n_samples
        df = pd.DataFrame({"road_name": road_names,
                           "base_rank": base_ranks[mth],
                           "holistic_acceptability": a_h,
                           "first_rank_acceptability": first / n_samples,
                           "top10_acceptability": b.sum(axis=1)})
        for r_ in range(n_rank):
            df["b{}".format(r_ + 1)] = b[:, r_]
        with np.errstate(invalid="ignore", divide="ignore"):
            cw = cw_sum / first[:, None]
        for j, c in enumerate(CRITERIA):
            df["central_w_{}".format(c)] = cw[:, j]
        df["smaa_rank"] = rankdata(-a_h, method="min")
        out[mth] = df.sort_values("smaa_rank").reset_index(drop=True)
        stats.append({"method": mth, "n_samples": n_samples,
                      "spearman_mean": float(rhos.mean()),
                      "spearman_p05": float(np.percentile(rhos, 5)),
                      "spearman_min": float(rhos.min()),
                      "spearman_base_vs_smaa": float(
                          spearmanr(base_ranks[mth], -a_h)[0]),
                      "top10_base_vs_smaa": topk_overlap(
                          base_ranks[mth], rankdata(-a_h), 10)})
    return out, pd.DataFrame(stats)


# ---------------------------------------------------------------------------
# 7. Per-city pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8. Paper-ready cross-city table
# ---------------------------------------------------------------------------
