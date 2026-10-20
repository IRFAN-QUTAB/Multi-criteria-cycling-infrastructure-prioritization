import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, kendalltau, pearsonr, rankdata, spearmanr



HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Main runs (physical-length routing cost, 25% closure) = paper results
CITY_FILES = {
    "modena": "modena_results/modena_20260711_005543_pct25.csv",
    "ferrara": "ferrara/ferrara_20260711_075849_pct25.csv",
    "pesaro": "pesaro/pesaro_20260711_084658_pct25.csv",
}
# Stress-weighted routing runs
CITY_FILES_STRESS = {
    "modena": "lts_runs/modena/modena_20260711_100449_pct25.csv",
    "ferrara": "lts_runs_fixed/ferrara/ferrara_20261015_135437_pct25.csv",
    "pesaro": "lts_runs/pesaro/pesaro_20260711_132848_pct25.csv",
}
# Closure-depth runs (available for Modena)
CITY_FILES_DEPTH = {
    "modena": {"10%": "sens_runs/modena 10%/modena_20260711_150454_pct10.csv",
               "50%": "sens_runs/modena 50%/modena_20260712_223616_pct50.csv"},
}
EXTERNAL_DIR = os.path.join(HERE, "external")

N_OD_PAIRS = 200

CRITERIA = ["C1_pagerank", "C2_modal_impact", "C3_poi_loss",
            "C4_fragmentation", "C5_compensation_failure"]
CRITERIA_LABELS = {
    "C1_pagerank": "Network importance (dual PageRank)",
    "C2_modal_impact": "Modal impact (max excess detour)",
    "C3_poi_loss": "Service loss (share of OD pairs losing access)",
    "C4_fragmentation": "Fragmentation (new connected components)",
    "C5_compensation_failure": "Compensation failure of cycling layer",
}
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

COMP_MAP = {"none": 1.0, "partial": 0.5, "full": 0.0,
            "no_impact": 0.0, "bike_only_vulnerable": 0.8}
BIKE_RELEVANT = {"none", "partial", "bike_only_vulnerable"}

CORE_METHODS = ["NR_TOPSIS", "VIKOR", "PROMETHEE_II"]
EXTRA_METHODS = ["TOPSIS", "WSM", "EDAS", "MABAC", "MARCOS"]
ALL_METHODS = CORE_METHODS + EXTRA_METHODS
WEIGHT_SCHEMES = ["critic", "entropy", "critic_entropy", "equal"]
TOP_K = (10, 20)
PROMETHEE_PREFERENCE = "gaussian"




def load_csv(path):
    """Read a result CSV (files were written with Windows encoding)."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode {}".format(path))


def _excess(df, col):
    return np.clip(df[col].astype(float).values - 1.0, 0.0, 1.0)


def _frag(df, mode):
    d = df["{}_wcc_delta".format(mode)].astype(float).values
    return np.where(d > 0, np.minimum(d / 100.0, 0.2), 0.0)


def build_decision_matrix(df, n_od=N_OD_PAIRS):
    """
    m x 5 decision matrix; each criterion reproduces one IPS term:
      C1  dual PageRank of the road
      C2  max over modes of min(max(detour - 1, 0), 1)
      C3  (car + bike_safe + bike_all POIs lost) / (3 * n_od)
      C4  sum over modes of min(wcc_delta / 100, 0.2) for wcc_delta > 0
      C5  max(COMP_MAP[bike_safe_compensates], COMP_MAP[bike_all_compensates])
    """
    out = pd.DataFrame({"road_name": df["road_name"].values,
                        "highway": df["highway"].values})
    out["C1_pagerank"] = df["pagerank"].astype(float).values
    car_imp = _excess(df, "car_avg_detour")
    bike_imp = np.maximum(_excess(df, "bike_safe_avg_detour"),
                          _excess(df, "bike_all_avg_detour"))
    out["C2_modal_impact"] = np.maximum(car_imp, bike_imp)
    lost = (df["car_pois_lost"].astype(float)
            + df["bike_safe_pois_lost"].astype(float)
            + df["bike_all_pois_lost"].astype(float))
    out["C3_poi_loss"] = (lost / max(n_od * 3, 1)).values
    out["C4_fragmentation"] = (_frag(df, "car") + _frag(df, "bike_safe")
                               + _frag(df, "bike_all"))
    bs = df["bike_safe_compensates"].map(COMP_MAP).fillna(0.0).values
    ba = df["bike_all_compensates"].map(COMP_MAP).fillna(0.0).values
    out["C5_compensation_failure"] = np.maximum(bs, ba)

    out["Ccar_impact"] = car_imp
    out["Ccar_poi_loss"] = (df["car_pois_lost"].astype(float) / n_od).values
    out["Ccar_fragmentation"] = _frag(df, "car")

    out["compensation_class"] = df["bike_safe_compensates"].values
    out["bike_relevant"] = out["compensation_class"].isin(BIKE_RELEVANT)
    out["bike_only_vulnerable"] = (out["compensation_class"]
                                   == "bike_only_vulnerable")
    out["IPS_original"] = df["investment_priority_score"].astype(float).values
    return out


def domain_bounds(X, criteria=CRITERIA):
    """[m_j, M_j] for NR-TOPSIS; observed range where no bound is defined."""
    lo, hi = X.min(axis=0).astype(float), X.max(axis=0).astype(float)
    for j, c in enumerate(criteria):
        if c in DOMAIN_BOUNDS:
            lo[j], hi[j] = DOMAIN_BOUNDS[c]
    return lo, hi




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


def setup_robustness(city, names, base_cons, primary):
    """Routing cost (all cities) and closure depth (Modena) robustness."""
    rows = []
    variants = {}
    if city in CITY_FILES_STRESS:
        variants["stress_weighted_cost"] = CITY_FILES_STRESS[city]
    for lab, rel in CITY_FILES_DEPTH.get(city, {}).items():
        variants["closure_depth_{}".format(lab)] = rel
    for label, rel in variants.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        dm_v = build_decision_matrix(load_csv(path)).set_index("road_name")
        common = [r for r in names if r in dm_v.index]
        idx = [list(names).index(r) for r in common]
        Xv = dm_v.loc[common, CRITERIA].values
        _, cons_v = core_consensus(Xv, WEIGHT_FUNCS[primary](Xv),
                                   None, domain_bounds(Xv))
        base_sub = rankdata(base_cons[idx], method="average")
        row = {"variant": label, "n_common_roads": len(common)}
        row.update({"spearman": spearmanr(base_sub, cons_v)[0],
                    "WS": ws_coefficient(base_sub, cons_v),
                    "top10_overlap": topk_overlap(base_sub, cons_v, 10),
                    "top20_overlap": topk_overlap(base_sub, cons_v, 20)})
        rows.append(row)
    return pd.DataFrame(rows)


def external_validation(city, ranks, road_names):
    """Optional: ESWA/external/<city>_planned_roads.csv (column road_name)."""
    path = os.path.join(EXTERNAL_DIR, "{}_planned_roads.csv".format(city))
    if not os.path.exists(path):
        return None
    planned = set(load_csv(path)["road_name"].astype(str).str.strip())
    is_planned = np.array([str(r).strip() in planned for r in road_names])
    M, Kp = len(road_names), int(is_planned.sum())
    rows = []
    for name, r in ranks.items():
        for k in TOP_K:
            hits = int(is_planned[list(top_set(r, k))].sum())
            rows.append({"ranking": name, "k": k, "planned_in_list": Kp,
                         "hits": hits, "precision_at_k": hits / k,
                         "p_value": float(hypergeom.sf(hits - 1, M, Kp, k))})
    return pd.DataFrame(rows)




def analyse_city(city, primary, smaa_samples, rr_iter, out_root):
    out_dir = os.path.join(out_root, city)
    os.makedirs(out_dir, exist_ok=True)
    tables = {}

    def save(df, name, key=None):
        df.to_csv(os.path.join(out_dir, name), index=False)
        if key:
            tables[key] = df

    dm = build_decision_matrix(load_csv(os.path.join(ROOT, CITY_FILES[city])))
    X = dm[CRITERIA].values
    names = dm["road_name"].values
    bounds = domain_bounds(X)
    save(dm, "decision_matrix.csv")
    save(dm[CRITERIA].describe().T.reset_index().rename(
        columns={"index": "criterion"}), "criteria_descriptives.csv")
    pd.DataFrame(np.corrcoef(X, rowvar=False), index=CRITERIA,
                 columns=CRITERIA).to_csv(
        os.path.join(out_dir, "criteria_correlation.csv"))

    # ---- weights ----
    weights = {k: f(X) for k, f in WEIGHT_FUNCS.items()}
    wdf = pd.DataFrame(weights, index=CRITERIA)
    wdf.index.name = "criterion"
    wdf.to_csv(os.path.join(out_dir, "weights.csv"))
    tables["weights"] = wdf.reset_index()
    w = weights[primary]
    flows = promethee_criterion_flows(X)

    # ---- rankings ----
    res = run_methods(X, w, ALL_METHODS, flows, bounds)
    base_ranks = {m_: res[m_][1] for m_ in ALL_METHODS}
    mean_r, cons_r = consensus_rank([base_ranks[m_] for m_ in CORE_METHODS],
                                    tiebreak=res["NR_TOPSIS"][0])

    Xc = dm[CAR_CRITERIA].values
    car_res, car_cons = core_consensus(Xc, WEIGHT_FUNCS[primary](Xc), None,
                                       domain_bounds(Xc, CAR_CRITERIA))
    benchmarks = {"PageRank_only": rank_from_score(dm["C1_pagerank"].values),
                  "CarOnly_MCDM": car_cons,
                  "IPS_original": rank_from_score(dm["IPS_original"].values)}

    scores = dm[["road_name", "highway", "compensation_class"] + CRITERIA].copy()
    for m_ in ALL_METHODS:
        scores["{}_score".format(m_)] = res[m_][0]
        scores["{}_rank".format(m_)] = res[m_][1]
    S, R, Q = vikor(X, w)
    scores["VIKOR_S"], scores["VIKOR_R"] = S, R
    scores["mean_core_rank"] = mean_r
    scores["consensus_rank"] = cons_r
    scores["IPS_original"] = dm["IPS_original"]
    for b, r in benchmarks.items():
        scores["{}_rank".format(b)] = r
    save(scores.sort_values("consensus_rank").reset_index(drop=True),
         "scores_rankings_{}.csv".format(primary))

    cond = vikor_conditions(S, R, Q)
    cond["compromise_set"] = [names[i] for i in cond.pop("compromise_set_idx")]

    # ---- (1) comparative analysis with other MCDM methods ----
    ranks_methods = dict(base_ranks)
    ranks_methods["Consensus"] = cons_r
    oriented = {m_: (res[m_][0] if HIGHER_IS_BETTER[m_] else -res[m_][0])
                for m_ in ALL_METHODS}
    save(compare_rankings(ranks_methods, oriented),
         "rank_similarity_methods.csv", "rank_similarity_methods")
    kw = {"core_methods": float(kendall_w([base_ranks[m_] for m_ in CORE_METHODS])),
          "all_methods": float(kendall_w([base_ranks[m_] for m_ in ALL_METHODS]))}
    tables["kendall_w"] = pd.DataFrame([kw])

    # ---- (2) weighting methods ----
    ranks_w = {}
    for ws in WEIGHT_SCHEMES:
        r_ws, c_ws = core_consensus(X, weights[ws], flows, bounds)
        for m_ in CORE_METHODS:
            ranks_w["{}|{}".format(m_, ws)] = r_ws[m_][1]
        ranks_w["Consensus|{}".format(ws)] = c_ws
    save(compare_rankings(ranks_w), "rank_similarity_weights.csv", "weight_schemes")

    # ---- (3) criteria-weight scenarios ----
    wsc, wsc_ranks = weight_scenarios(X, w, base_ranks, cons_r, flows, bounds)
    save(wsc, "sensitivity_weight_scenarios.csv", "weight_scenarios")
    top10_idx = np.argsort(cons_r, kind="mergesort")[:10]
    traj = pd.DataFrame({"road_name": names[top10_idx], "base": cons_r[top10_idx]})
    for sid, rr in wsc_ranks:
        traj[sid] = rr[top10_idx]
    save(traj, "sensitivity_weight_scenarios_top10_ranks.csv")
    save(sensitivity_oat(X, w, base_ranks, flows, bounds),
         "sensitivity_oat.csv", "sensitivity_oat")

    # ---- (4) criteria-group scenarios ----
    save(criteria_group_scenarios(X, w, cons_r, bounds),
         "sensitivity_criteria_groups.csv", "criteria_groups")

    # ---- (5) rank reversal ----
    save(rank_reversal_worst_deletion(X, w, bounds),
         "rank_reversal_worst_deletion.csv", "rank_reversal_worst")
    save(rank_reversal_random(X, w, bounds, n_iter=rr_iter),
         "rank_reversal_random.csv", "rank_reversal_random")

    # ---- (6) method parameters ----
    save(pd.DataFrame(vikor_v_scenarios(X, w, base_ranks["VIKOR"])
                      + promethee_scenarios(X, w, base_ranks["PROMETHEE_II"])),
         "sensitivity_method_parameters.csv", "method_parameters")

    # ---- (7) SMAA-2 ----
    smaa_tables, smaa_stats = smaa2(X, names, flows, bounds, base_ranks,
                                    n_samples=smaa_samples)
    for m_, t in smaa_tables.items():
        save(t, "smaa_{}.csv".format(m_))
    save(smaa_stats, "smaa_summary.csv", "smaa")

    # ---- (8) comprehensiveness ----
    comp_scores = {m_: oriented[m_] for m_ in CORE_METHODS}
    comp_scores["Consensus"] = -cons_r
    save(comprehensiveness(X, w, bounds, comp_scores),
         "criteria_contribution.csv", "criteria_contribution")

    # ---- (9) benchmarks ----
    ranks_bench = {"Consensus": cons_r}
    ranks_bench.update(benchmarks)
    save(compare_rankings(ranks_bench), "benchmark_similarity.csv",
         "benchmark_similarity")
    all_r = dict(ranks_methods)
    all_r.update(benchmarks)
    all_s = dict(oriented)
    all_s.update({"Consensus": -cons_r,
                  "PageRank_only": dm["C1_pagerank"].values,
                  "IPS_original": dm["IPS_original"].values,
                  "CarOnly_MCDM": -car_cons})
    prof, disc = [], []
    for name, r in all_r.items():
        for k in TOP_K:
            row = {"ranking": name}
            row.update(topk_profile(r, dm, k))
            prof.append(row)
        d = {"ranking": name}
        d.update(discrimination(all_s[name]))
        disc.append(d)
    save(pd.DataFrame(prof), "topk_profile.csv", "topk_profile")
    save(pd.DataFrame(disc), "discrimination_power.csv", "discrimination_power")
    only = sorted(top_set(cons_r, 20) - top_set(car_cons, 20),
                  key=lambda i: cons_r[i])
    save(dm.iloc[only][["road_name", "highway", "compensation_class"]
                       + CRITERIA].assign(consensus_rank=cons_r[only],
                                          car_only_rank=car_cons[only]),
         "top20_missed_by_car_assessment.csv")

    # ---- (10) simulation set-up ----
    save(setup_robustness(city, names, cons_r, primary),
         "robustness_simulation_setup.csv", "robustness_setup")

    # ---- (11) external validation ----
    ext = external_validation(city, all_r, names)
    if ext is not None:
        save(ext, "external_validation.csv", "external_validation")

    summary = {
        "city": city, "n_alternatives": int(len(dm)),
        "primary_weights": primary,
        "weights": {k: dict(zip(CRITERIA, np.round(v_, 4).tolist()))
                    for k, v_ in weights.items()},
        "nr_topsis_domain_bounds": {c: [float(bounds[0][j]), float(bounds[1][j])]
                                    for j, c in enumerate(CRITERIA)},
        "kendall_W": kw,
        "vikor_conditions": cond,
        "top10_consensus": [names[i] for i in
                            np.argsort(cons_r, kind="mergesort")[:10]],
        "n_bike_only_vulnerable": int(dm["bike_only_vulnerable"].sum()),
        "external_validation": ext is not None,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, tables




def paper_table(all_tables, cities):
    rows = []
    for city in cities:
        t = all_tables[city]
        sim = t["rank_similarity_methods"]
        core = sim[sim.ranking_A.isin(CORE_METHODS) & sim.ranking_B.isin(CORE_METHODS)]
        allm = sim[sim.ranking_A.isin(ALL_METHODS) & sim.ranking_B.isin(ALL_METHODS)]
        bench = t["benchmark_similarity"]
        bench = bench[bench.ranking_A == "Consensus"].set_index("ranking_B")
        p10 = t["topk_profile"][t["topk_profile"].k == 10].set_index("ranking")
        p20 = t["topk_profile"][t["topk_profile"].k == 20].set_index("ranking")
        disc = t["discrimination_power"].set_index("ranking")
        rrw = t["rank_reversal_worst"].set_index("method")
        rrr = t["rank_reversal_random"].set_index("method")
        wsc = t["weight_scenarios"]
        wsc_c = wsc[wsc.method == "Consensus"]
        grp = t["criteria_groups"]
        smaa = t["smaa"].set_index("method")
        rob = t["robustness_setup"].set_index("variant") if len(t["robustness_setup"]) else None
        row = {
            "city": city,
            "kendall_W_core": t["kendall_w"]["core_methods"].iloc[0],
            "kendall_W_all": t["kendall_w"]["all_methods"].iloc[0],
            "core_spearman_min": core.spearman_rho.min(),
            "all_methods_spearman_min": allm.spearman_rho.min(),
            "core_top10_overlap_min": core.top10_overlap.min(),
            "consensus_vs_IPS_top10": bench.loc["IPS_original", "top10_overlap"],
            "consensus_vs_carOnly_top10": bench.loc["CarOnly_MCDM", "top10_overlap"],
            "consensus_vs_pagerank_top10": bench.loc["PageRank_only", "top10_overlap"],
            "bike_relevant_top20_consensus": p20.loc["Consensus", "bike_relevant_share"],
            "bike_relevant_top20_carOnly": p20.loc["CarOnly_MCDM", "bike_relevant_share"],
            "bike_relevant_top20_pagerank": p20.loc["PageRank_only", "bike_relevant_share"],
            "bike_relevant_top20_IPS": p20.loc["IPS_original", "bike_relevant_share"],
            "bike_relevant_top10_consensus": p10.loc["Consensus", "bike_relevant_share"],
            "tied_share_IPS": disc.loc["IPS_original", "tied_share"],
            "tied_share_consensus": disc.loc["Consensus", "tied_share"],
            "weight_scen_top1_unchanged_share": wsc_c.top1_unchanged.mean(),
            "weight_scen_top10_retained_min": wsc_c.top10_retained.min(),
            "weight_scen_spearman_min": wsc_c.spearman_vs_base.min(),
            "criteria_groups_spearman_min": grp.spearman_vs_base.min(),
            "rr_worst_steps_with_reversal_NR_TOPSIS":
                rrw.loc["NR_TOPSIS", "steps_with_any_reversal"],
            "rr_worst_steps_with_reversal_TOPSIS":
                rrw.loc["TOPSIS", "steps_with_any_reversal"],
            "rr_worst_steps": rrw.loc["NR_TOPSIS", "steps"],
            "rr_random_top10_NR_TOPSIS": rrr.loc["NR_TOPSIS", "top10_retained_mean"],
            "smaa_spearman_p05_NR_TOPSIS": smaa.loc["NR_TOPSIS", "spearman_p05"],
        }
        if rob is not None:
            for v in rob.index:
                row["{}_top10".format(v)] = rob.loc[v, "top10_overlap"]
                row["{}_spearman".format(v)] = rob.loc[v, "spearman"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="MCDM road prioritisation")
    ap.add_argument("--primary-weights", default="critic", choices=WEIGHT_SCHEMES)
    ap.add_argument("--promethee-preference", default="gaussian",
                    choices=["gaussian", "vshape"])
    ap.add_argument("--smaa-samples", type=int, default=2000)
    ap.add_argument("--rr-iter", type=int, default=200)
    ap.add_argument("--cities", nargs="+", default=list(CITY_FILES.keys()))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    args = ap.parse_args()

    global PROMETHEE_PREFERENCE
    PROMETHEE_PREFERENCE = args.promethee_preference
    os.makedirs(args.out, exist_ok=True)
    all_tables = {}
    for city in args.cities:
        print("=== {} ===".format(city))
        summary, tables = analyse_city(city, args.primary_weights,
                                       args.smaa_samples, args.rr_iter, args.out)
        all_tables[city] = tables
        print("weights:", summary["weights"][args.primary_weights])
        print("Kendall W:", {k: round(v, 4) for k, v in summary["kendall_W"].items()})
        print("Top-10 consensus:", summary["top10_consensus"])
        print()

    keys = set().union(*[set(t.keys()) for t in all_tables.values()])
    for key in sorted(keys):
        parts = []
        for city, t in all_tables.items():
            if key in t:
                d = t[key].copy()
                d.insert(0, "city", city)
                parts.append(d)
        pd.concat(parts).to_csv(
            os.path.join(args.out, "summary_{}.csv".format(key)), index=False)

    pt = paper_table(all_tables, args.cities)
    pt.to_csv(os.path.join(args.out, "summary_paper_table.csv"), index=False)
    print(pt.round(3).T.to_string())
    print("\nOutputs written to", args.out)


if __name__ == "__main__":
    main()
