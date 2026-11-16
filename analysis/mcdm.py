
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from metrics import (OUTPUTS_DIR, RESULTS_DIR, Variant,  # noqa
                     compensation_class, scenario_metrics, wide)

_spec = importlib.util.spec_from_file_location(
    "mcdm_methods", os.path.join(HERE, "mcdm_methods.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

CRITERIA = M.CRITERIA
CORE, ALL = M.CORE_METHODS, M.ALL_METHODS
CAR_CRITERIA = M.CAR_CRITERIA
BETA = {"none": 1.0, "partial": 0.5, "full": 0.0, "no_impact": 0.0,
        "bike_only_vulnerable": 0.8}
BETA_ALT = {
    "bov_1.0": dict(BETA, bike_only_vulnerable=1.0),
    "bov_0.6": dict(BETA, bike_only_vulnerable=0.6),
    "partial_0.3": dict(BETA, partial=0.3),
    "partial_0.7": dict(BETA, partial=0.7),
    "binary": {"none": 1.0, "partial": 1.0, "full": 0.0, "no_impact": 0.0,
               "bike_only_vulnerable": 1.0},
}
BIKE_RELEVANT = {"none", "partial", "bike_only_vulnerable"}
SETUP_VARIANTS = ["depth10", "depth50", "depth100", "cost_lts", "lts_stored",
                  "lts_zone30", "access_legacy", "dest_layer"]
TOP_K = (10, 20)


# ---------------------------------------------------------------------------
def decision_matrix(var, alts, sets=("A",), rule="iso",
                    layers=("bike_safe",), beta=BETA, pair_ids=None,
                    tol=0.01, c5="excess"):
    """
    C5 variants:
      excess   (main) uncompensated low-stress harm, max(0, H_safe - H_car),
               with H = adjusted detour - 1 (population-weighted, valid
               pairs): the part of the low-stress loss that the car layer
               does not suffer; continuous, no scores or tolerance
      share    H_safe / (H_safe + H_car)
      ordinal  beta score of the compensation class (first version)
    """
    met = scenario_metrics(var, sets=sets, rule=rule, pair_ids=pair_ids)
    w = wide(met)
    sc = var.scen[["scenario", "road_name", "pagerank"]]
    df = sc.merge(w, on="scenario", how="inner")
    df = df.merge(alts[["road_name", "highway_mode"]], on="road_name",
                  how="left").rename(columns={"highway_mode": "highway"})
    for c in [c for c in df.columns if c.startswith("cut_")]:
        df[c] = df[c].fillna(0.0)
    for c in [c for c in df.columns if c.split("_")[0] in ("L", "e", "adj")]:
        df[c] = df[c].fillna(0.0 if not c.startswith("adj") else 1.0)
    out = pd.DataFrame({"road_name": df["road_name"].values,
                        "highway": df["highway"].values})
    out["C1_pagerank"] = df["pagerank"].astype(float).values
    out["C2_modal_impact"] = np.max([df["e_" + m].values for m in layers],
                                    axis=0)
    out["C3_poi_loss"] = np.max([df["L_" + m].values for m in layers], axis=0)
    out["C4_fragmentation"] = np.max([df["cut_" + m].values for m in layers],
                                     axis=0)
    cls = {}
    for m in ("bike_safe", "bike_all"):
        cls[m] = compensation_class(df["L_car"].values, df["e_car"].values,
                                    df["L_" + m].values, df["e_" + m].values,
                                    tol)
    Hc = np.clip(df["adj_car"].values - 1.0, 0, None)
    if c5 == "excess":
        v = [np.clip(np.clip(df["adj_" + m].values - 1.0, 0, None) - Hc,
                     0, None) for m in layers]
    elif c5 == "share":
        v = []
        for m in layers:
            Hm = np.clip(df["adj_" + m].values - 1.0, 0, None)
            den = Hm + Hc
            v.append(np.where(den > 0, Hm / np.where(den > 0, den, 1), 0.0))
    else:
        v = [np.vectorize(beta.get)(cls[m]) for m in layers]
    out["C5_compensation_failure"] = np.max(v, axis=0)
    out["Ccar_impact"] = df["e_car"].values
    out["Ccar_poi_loss"] = df["L_car"].values
    out["Ccar_fragmentation"] = df["cut_car"].values
    out["compensation_class"] = cls["bike_safe"]
    out["compensation_class_all"] = cls["bike_all"]
    out["bike_relevant"] = out["compensation_class"].isin(BIKE_RELEVANT)
    out["bike_only_vulnerable"] = (out["compensation_class"]
                                   == "bike_only_vulnerable")
    out["IPS_original"] = (out["C1_pagerank"]
                           * (out["C2_modal_impact"] + out["C3_poi_loss"]
                              + out["C4_fragmentation"])
                           * (0.5 + out["C5_compensation_failure"]))
    for m in ("car", "bike_all", "bike_safe"):
        for k in ("L", "e", "adj", "legacy_adj", "cut", "n_excl"):
            if "{}_{}".format(k, m) in df.columns:
                out["{}_{}".format(k, m)] = df["{}_{}".format(k, m)].values
    return out.sort_values("road_name").reset_index(drop=True)


def fixed_bounds(X):
    X = np.asarray(X, float)
    lo = np.zeros(X.shape[1])
    hi = X.max(axis=0).astype(float)
    lo[0] = X[:, 0].min()
    hi = np.where(hi - lo <= 0, lo + 1.0, hi)
    return lo, hi


def consensus_of(X, w, bounds):
    res, cons = M.core_consensus(X, w, None, bounds)
    return res, cons


def ranking_from_dm(dm, primary="critic"):
    X = dm[CRITERIA].values
    w = M.WEIGHT_FUNCS[primary](X)
    _, cons = consensus_of(X, w, fixed_bounds(X))
    return cons


def compare_common(names_a, rank_a, names_b, rank_b):
    a = pd.Series(rank_a, index=names_a)
    b = pd.Series(rank_b, index=names_b)
    common = sorted(set(a.index) & set(b.index))
    ra = rankdata(a.loc[common].values)
    rb = rankdata(b.loc[common].values)
    return {"n_common": len(common),
            "spearman": spearmanr(ra, rb)[0],
            "kendall": kendalltau(ra, rb)[0],
            "WS": M.ws_coefficient(ra, rb),
            "top10_overlap": M.topk_overlap(ra, rb, 10),
            "top20_overlap": M.topk_overlap(ra, rb, 20),
            "top1_same": bool(np.argmin(ra) == np.argmin(rb))}


def rank_reversal_reestimated(X, primary, max_share=0.5, methods=ALL):
    """Delete the worst road repeatedly; weights (CRITIC) and NR-TOPSIS
    bounds are re-estimated on the remaining set at every step, as in a
    real re-application of the framework."""
    m = X.shape[0]
    n_steps = int(max_share * m)
    rows = []
    for mth in methods:
        w0 = M.WEIGHT_FUNCS[primary](X)
        full = M.run_methods(X, w0, [mth], bounds=fixed_bounds(X))[mth][1]
        keep = np.arange(m)
        cur = full.copy()
        any_ch, top_ch, taus = 0, 0, []
        for _ in range(n_steps):
            worst = keep[np.argmax(cur)]
            keep = keep[keep != worst]
            Xk = X[keep]
            wk = M.WEIGHT_FUNCS[primary](Xk)
            new = M.run_methods(Xk, wk, [mth],
                                  bounds=fixed_bounds(Xk))[mth][1]
            ref = rankdata(full[keep], method="average")
            tau = kendalltau(ref, new)[0]
            taus.append(tau)
            any_ch += tau < 1 - 1e-12
            top_ch += (list(np.argsort(ref, kind="mergesort")[:10])
                       != list(np.argsort(new, kind="mergesort")[:10]))
            cur = new
        rows.append({"method": mth, "steps": n_steps,
                     "steps_with_any_reversal": int(any_ch),
                     "steps_with_top10_order_change": int(top_ch),
                     "kendall_tau_min": float(np.min(taus)),
                     "kendall_tau_mean": float(np.mean(taus))})
    return pd.DataFrame(rows)


def paired_wilcoxon(x, y):
    """Wilcoxon signed-rank with matched-pairs rank-biserial effect size."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 5:
        return {"n_nonzero": int(len(d)), "W": np.nan, "p": np.nan,
                "rank_biserial": np.nan, "median_diff": np.nan}
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    res = wilcoxon(d)
    return {"n_nonzero": int(len(d)), "W": float(res.statistic),
            "p": float(res.pvalue),
            "rank_biserial": float((wp - wm) / (wp + wm)),
            "median_diff": float(np.median(d))}


# ---------------------------------------------------------------------------
def analyse_city(city, args):
    res_dir = os.path.join(BASE, RESULTS_DIR, city)
    out_dir = os.path.join(BASE, OUTPUTS_DIR, city)
    os.makedirs(out_dir, exist_ok=True)
    T = {}

    def save(df, name, key=None):
        df.to_csv(os.path.join(out_dir, name), index=False)
        if key:
            T[key] = df

    alts = pd.read_csv(os.path.join(res_dir, "alternatives.csv"))
    main = Variant(os.path.join(res_dir, "single_main"))
    dm = decision_matrix(main, alts, sets=("A",))
    X = dm[CRITERIA].values
    names = dm["road_name"].values
    bounds = fixed_bounds(X)
    save(dm, "decision_matrix.csv")
    save(dm[CRITERIA].describe().T.reset_index().rename(
        columns={"index": "criterion"}), "criteria_descriptives.csv")
    zero = {c: float((dm[c] == 0).mean()) for c in CRITERIA}
    pd.DataFrame(np.corrcoef(X, rowvar=False), index=CRITERIA,
                 columns=CRITERIA).to_csv(
        os.path.join(out_dir, "criteria_correlation.csv"))

    weights = {k: f(X) for k, f in M.WEIGHT_FUNCS.items()}
    wdf = pd.DataFrame(weights, index=CRITERIA)
    wdf.index.name = "criterion"
    wdf.to_csv(os.path.join(out_dir, "weights.csv"))
    T["weights"] = wdf.reset_index()
    w = weights[args.primary]
    flows = M.promethee_criterion_flows(X)

    res = M.run_methods(X, w, ALL, flows, bounds)
    base_ranks = {m: res[m][1] for m in ALL}
    mean_r, cons = M.consensus_rank([base_ranks[m] for m in CORE],
                                      tiebreak=res["NR_TOPSIS"][0])
    Xc = dm[CAR_CRITERIA].values
    _, car_cons = consensus_of(Xc, M.WEIGHT_FUNCS[args.primary](Xc),
                               fixed_bounds(Xc))
    bench = {"PageRank_only": M.rank_from_score(dm["C1_pagerank"].values),
             "CarOnly_MCDM": car_cons,
             "IPS_original": M.rank_from_score(dm["IPS_original"].values)}

    sc = dm.copy()
    for m in ALL:
        sc[m + "_score"] = res[m][0]
        sc[m + "_rank"] = res[m][1]
    S, R, Q = M.vikor(X, w)
    sc["VIKOR_S"], sc["VIKOR_R"] = S, R
    sc["mean_core_rank"] = mean_r
    sc["consensus_rank"] = cons
    for b, r in bench.items():
        sc[b + "_rank"] = r
    save(sc.sort_values("consensus_rank").reset_index(drop=True),
         "scores_rankings.csv", "scores")
    cond = M.vikor_conditions(S, R, Q)
    cond["compromise_set"] = [names[i] for i in cond.pop("compromise_set_idx")]

    # comparative analysis
    ranks_m = dict(base_ranks)
    ranks_m["Consensus"] = cons
    oriented = {m: (res[m][0] if M.HIGHER_IS_BETTER[m] else -res[m][0])
                for m in ALL}
    save(M.compare_rankings(ranks_m, oriented),
         "rank_similarity_methods.csv", "rank_similarity_methods")
    kw = {"core_methods": float(M.kendall_w([base_ranks[m] for m in CORE])),
          "all_methods": float(M.kendall_w([base_ranks[m] for m in ALL]))}
    T["kendall_w"] = pd.DataFrame([kw])

    # weighting methods
    ranks_w = {}
    for ws in M.WEIGHT_SCHEMES:
        r_ws, c_ws = M.core_consensus(X, weights[ws], flows, bounds)
        for m in CORE:
            ranks_w["{}|{}".format(m, ws)] = r_ws[m][1]
        ranks_w["Consensus|{}".format(ws)] = c_ws
    save(M.compare_rankings(ranks_w), "rank_similarity_weights.csv",
         "weight_schemes")

    # weight scenarios, OAT, criteria groups
    wsc, wsc_ranks = M.weight_scenarios(X, w, base_ranks, cons, flows,
                                          bounds)
    save(wsc, "sensitivity_weight_scenarios.csv", "weight_scenarios")
    top10 = np.argsort(cons, kind="mergesort")[:10]
    traj = pd.DataFrame({"road_name": names[top10], "base": cons[top10]})
    for sid, rr in wsc_ranks:
        traj[sid] = rr[top10]
    save(traj, "sensitivity_weight_scenarios_top10_ranks.csv")
    save(M.sensitivity_oat(X, w, base_ranks, flows, bounds),
         "sensitivity_oat.csv", "sensitivity_oat")
    save(M.criteria_group_scenarios(X, w, cons, bounds),
         "sensitivity_criteria_groups.csv", "criteria_groups")

    # rank reversal: fixed (implementation check) and re-estimated
    save(M.rank_reversal_worst_deletion(X, w, bounds),
         "rank_reversal_worst_fixed.csv", "rr_fixed")
    save(M.rank_reversal_random(X, w, bounds, n_iter=args.rr_iter),
         "rank_reversal_random_fixed.csv", "rr_random_fixed")
    save(rank_reversal_reestimated(X, args.primary),
         "rank_reversal_worst_reestimated.csv", "rr_reest")

    # method parameters, SMAA, contribution
    save(pd.DataFrame(M.vikor_v_scenarios(X, w, base_ranks["VIKOR"])
                      + M.promethee_scenarios(X, w,
                                                base_ranks["PROMETHEE_II"])),
         "sensitivity_method_parameters.csv", "method_parameters")
    smaa_t, smaa_s = M.smaa2(X, names, flows, bounds, base_ranks,
                               n_samples=args.smaa_samples)
    for m, t in smaa_t.items():
        save(t, "smaa_{}.csv".format(m))
    save(smaa_s, "smaa_summary.csv", "smaa")
    cs = {m: oriented[m] for m in CORE}
    cs["Consensus"] = -cons
    save(M.comprehensiveness(X, w, bounds, cs),
         "criteria_contribution.csv", "criteria_contribution")

    # benchmarks
    rb = {"Consensus": cons}
    rb.update(bench)
    save(M.compare_rankings(rb), "benchmark_similarity.csv",
         "benchmark_similarity")
    allr = dict(ranks_m)
    allr.update(bench)
    alls = dict(oriented)
    alls.update({"Consensus": -cons, "PageRank_only": dm["C1_pagerank"].values,
                 "IPS_original": dm["IPS_original"].values,
                 "CarOnly_MCDM": -car_cons})
    prof, disc = [], []
    for nm, r in allr.items():
        for k in TOP_K:
            row = {"ranking": nm}
            row.update(M.topk_profile(r, dm, k))
            prof.append(row)
        d = {"ranking": nm}
        d.update(M.discrimination(alls[nm]))
        disc.append(d)
    save(pd.DataFrame(prof), "topk_profile.csv", "topk_profile")
    save(pd.DataFrame(disc), "discrimination_power.csv", "discrimination")
    only = sorted(M.top_set(cons, 20) - M.top_set(car_cons, 20),
                  key=lambda i: cons[i])
    save(dm.iloc[only][["road_name", "highway", "compensation_class"]
                       + CRITERIA].assign(consensus_rank=cons[only],
                                          car_only_rank=car_cons[only]),
         "top20_missed_by_car_assessment.csv")
    ranking_table = pd.DataFrame({"road_name": names, "Consensus": cons})
    for m in ALL:
        ranking_table[m] = base_ranks[m]
    for b, r in bench.items():
        ranking_table[b] = r
    save(ranking_table, "rankings_for_validation.csv")

    # -------- new sensitivity analyses --------
    rows = []
    for c5v in ("ordinal", "share"):
        d2 = decision_matrix(main, alts, sets=("A",), c5=c5v)
        rows.append(dict(scenario="C5_" + c5v, **compare_common(
            names, cons, d2["road_name"].values, ranking_from_dm(d2))))
    for lab, beta in BETA_ALT.items():
        d2 = decision_matrix(main, alts, sets=("A",), beta=beta, c5="ordinal")
        rows.append(dict(scenario="C5_ordinal_beta_" + lab, **compare_common(
            names, cons, d2["road_name"].values, ranking_from_dm(d2))))
    d2 = decision_matrix(main, alts, sets=("A",),
                         layers=("bike_safe", "bike_all"))
    rows.append(dict(scenario="criteria_on_both_cycling_layers",
                     **compare_common(names, cons, d2["road_name"].values,
                                      ranking_from_dm(d2))))
    for rule in ("inc", "none"):
        d2 = decision_matrix(main, alts, sets=("A",), rule=rule)
        rows.append(dict(scenario="exclusion_rule_" + rule,
                         **compare_common(names, cons,
                                          d2["road_name"].values,
                                          ranking_from_dm(d2))))
    bpath = os.path.join(BASE, OUTPUTS_DIR, city, "road_boundary.csv")
    if os.path.exists(bpath):
        bd = pd.read_csv(bpath)
        inner = set(bd.loc[bd["dist_to_bbox_edge_m"] >= 500, "road_name"])
        keep = np.array([n in inner for n in names])
        if keep.sum() > 20:
            Xi = X[keep]
            _, ci = consensus_of(Xi, M.WEIGHT_FUNCS[args.primary](Xi),
                                 fixed_bounds(Xi))
            rows.append(dict(scenario="without_roads_within_500m_of_edge",
                             **compare_common(names, cons, names[keep], ci)))
    save(pd.DataFrame(rows), "sensitivity_model_choices.csv",
         "model_choices")

    rows = []
    for v in SETUP_VARIANTS:
        vd = os.path.join(res_dir, "single_" + v)
        if not os.path.exists(os.path.join(vd, "done.txt")):
            continue
        dv = decision_matrix(Variant(vd), alts, sets=("A",))
        rows.append(dict(variant=v, **compare_common(
            names, cons, dv["road_name"].values, ranking_from_dm(dv))))
    save(pd.DataFrame(rows), "robustness_simulation_setup.csv",
         "robustness_setup")

    # OD-sample stability
    rows = []
    dB = decision_matrix(main, alts, sets=("B",))
    rows.append(dict(sample="set_B_vs_set_A", **compare_common(
        names, cons, dB["road_name"].values, ranking_from_dm(dB))))
    pa = main.pairs.loc[main.pairs["set"] == "A", "pair"].values
    rng = np.random.default_rng(7)
    perm = rng.permutation(pa)
    subs = [perm[i::5] for i in range(5)]
    sub_ranks = []
    for i, ids in enumerate(subs):
        ds = decision_matrix(main, alts, pair_ids=ids)
        rs = ranking_from_dm(ds)
        sub_ranks.append((ds["road_name"].values, rs))
        rows.append(dict(sample="subset{}_of_A_vs_A".format(i + 1),
                         n_pairs=len(ids), **compare_common(
                             names, cons, ds["road_name"].values, rs)))
    for i in range(5):
        for j in range(i + 1, 5):
            rows.append(dict(sample="subset{}_vs_subset{}".format(i + 1,
                                                                  j + 1),
                             **compare_common(*sub_ranks[i], *sub_ranks[j])))
    save(pd.DataFrame(rows), "robustness_od_sample.csv", "od_sample")

    # paired Wilcoxon: car vs low-stress per road (across roads)
    wrows = []
    for metric in ("adj", "L", "e"):
        r = paired_wilcoxon(dm["{}_car".format(metric)],
                            dm["{}_bike_safe".format(metric)])
        wrows.append(dict(metric=metric, comparison="car_minus_low_stress",
                          **r))
    save(pd.DataFrame(wrows), "wilcoxon_car_vs_lowstress_roads.csv",
         "wilcoxon")
    cls_counts = (dm.groupby(["compensation_class"]).size()
                  .rename("n").reset_index())
    save(cls_counts, "compensation_classes.csv", "classes")

    summary = {
        "city": city, "n_alternatives": int(len(dm)),
        "weights": {k: dict(zip(CRITERIA, np.round(v, 4).tolist()))
                    for k, v in weights.items()},
        "bounds": {c: [float(bounds[0][j]), float(bounds[1][j])]
                   for j, c in enumerate(CRITERIA)},
        "zero_share": zero, "kendall_W": kw, "vikor_conditions": cond,
        "top10_consensus": [names[i] for i in
                            np.argsort(cons, kind="mergesort")[:10]],
        "classes": cls_counts.set_index("compensation_class")["n"].to_dict(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+",
                    default=["modena", "ferrara", "pesaro"])
    ap.add_argument("--primary", default="critic")
    ap.add_argument("--smaa-samples", type=int, default=2000)
    ap.add_argument("--rr-iter", type=int, default=200)
    args = ap.parse_args()
    allT = {}
    for c in args.cities:
        if not os.path.exists(os.path.join(BASE, RESULTS_DIR, c,
                                           "single_main", "done.txt")):
            print("skip", c, "(no results)")
            continue
        print("=== {} ===".format(c))
        s, T = analyse_city(c, args)
        allT[c] = T
        print(json.dumps({k: s[k] for k in ("weights", "top10_consensus",
                                             "classes")},
                         indent=1, ensure_ascii=False, default=str)[:1500])
    out = os.path.join(BASE, OUTPUTS_DIR)
    keys = set().union(*[set(t) for t in allT.values()]) if allT else set()
    for key in sorted(keys):
        parts = []
        for c, t in allT.items():
            if key in t:
                d = t[key].copy()
                d.insert(0, "city", c)
                parts.append(d)
        pd.concat(parts).to_csv(os.path.join(out, "summary_{}.csv".format(
            key)), index=False)


if __name__ == "__main__":
    main()
