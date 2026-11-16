"""
sim_checks.py - simulation analyses that need the routing engine
(run after mcdm.py):

  stats     network statistics, coverage, connectivity, route overlap
  asym      targeted vs random simultaneous closures; trips inside the
            closure excluded; penalty sensitivity
  validate  top-k roads of each ranking closed together, measured on the
            held-out OD set B, k = 5..50, against random selections, with
            paired bootstrap confidence intervals
  invest    the top-k roads of each ranking are upgraded to LTS 1 and the
            gain in low-stress access is measured
  bypass    a parallel LTS 1 route is added along each selected road, the
            road is closed, and the avoided low-stress harm is measured

Usage: python analysis/sim_checks.py --city modena
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
from mmsim import closures as cl  # noqa: E402
from mmsim.citygraph import CityGraph  # noqa: E402
from mmsim.network import (LAYERS, Core, Layer, Simulator,  # noqa: E402
                           route_pairs)
from metrics import (OUTPUTS_DIR, RESULTS_DIR, Variant,  # noqa: E402
                     scenario_metrics)

P = 5.0
K_VAL = (5, 10, 20, 30, 50)
K_INV = (5, 10, 20)
RANKINGS_VAL = ["Consensus", "NR_TOPSIS", "VIKOR", "PROMETHEE_II",
                "CarOnly_MCDM", "PageRank_only", "IPS_original"]
N_BOOT = 2000


def load_sim(vdir):
    core = Core.load(os.path.join(vdir, "core.npz"))
    z = np.load(os.path.join(vdir, "access.npz"))
    acc = {m: (z["o_" + m], z["d_" + m]) for m in LAYERS}
    with open(os.path.join(vdir, "baseline.pkl"), "rb") as f:
        base = pickle.load(f)
    with open(os.path.join(vdir, "meta.json")) as f:
        meta = json.load(f)
    return Simulator(core, acc, meta["cost"], baseline=base), z["pair_ids"]


def pair_rt(sim, pair_ids, sets_of, rows, layer, sets, rule="iso",
            weights=None):
    """Per-pair r~ and weight (valid pairs of the given sets)."""
    b = sim.dist0[layer]
    sel = np.isin(sets_of, list(sets)) & np.isfinite(b) & (b > 0)
    rt = np.ones(len(b))
    excl = np.zeros(len(b), dtype=bool)
    for (m, i, dv, iso, inc) in rows:
        if m != layer:
            continue
        if (rule == "iso" and iso) or (rule == "inc" and inc):
            excl[i] = True
        rt[i] = P if not np.isfinite(dv) else dv / b[i] if b[i] > 0 else 1.0
    keep = sel & ~excl
    w = np.ones(len(b)) if weights is None else weights
    return pd.DataFrame({"rt": rt[keep], "w": w[keep]},
                        index=pair_ids[keep])


def summarize(d):
    if len(d) == 0:
        return {"n_valid": 0, "L": np.nan, "e": np.nan, "adj": np.nan}
    lost = d["rt"].values == P
    w = d["w"].values
    rt = d["rt"].values
    ws = w[~lost].sum()
    return {"n_valid": int(len(d)),
            "L": float(w[lost].sum() / w.sum()),
            "e": float((w[~lost] * (rt[~lost] - 1)).sum() / ws) if ws > 0
            else 0.0,
            "adj": float((w * rt).sum() / w.sum())}


def pair_weights(vdir, pair_ids):
    pairs = pd.read_csv(os.path.join(vdir, "pairs.csv")).set_index("pair")
    if "weight" in pairs.columns:
        return pairs.loc[pair_ids, "weight"].astype(float).values
    return np.ones(len(pair_ids))


# ---------------------------------------------------------------------------
def do_asym(city, out):
    var = Variant(os.path.join(BASE, RESULTS_DIR, city, "asym"))
    rows = []
    for rule in ("iso", "none"):
        met = scenario_metrics(var, sets=("A", "B"), rule=rule)
        met = met.merge(var.scen[["scenario", "kind", "k", "rep", "pool"]],
                        on="scenario")
        met["rule"] = rule
        rows.append(met)
    met = pd.concat(rows, ignore_index=True)
    met.to_csv(os.path.join(out, "asym_scenarios.csv"), index=False)
    g = []
    for (rule, layer, k), d in met.groupby(["rule", "layer", "k"]):
        t = d[d.kind == "targeted_pagerank"]
        row = {"rule": rule, "layer": layer, "k": k}
        for col in ("L", "e", "adj", "legacy_adj", "cut", "n_excl"):
            row["targeted_" + col] = float(t[col].iloc[0]) if len(t) else np.nan
            for pool in ("all_named", "alternatives"):
                r = d[(d.kind == "random") & (d.pool == pool)][col]
                row["{}_{}_mean".format(pool, col)] = float(r.mean())
                row["{}_{}_sd".format(pool, col)] = float(r.std(ddof=1))
                if col in ("adj", "legacy_adj"):
                    row["{}_{}_share_ge_targeted".format(pool, col)] = float(
                        (r >= row["targeted_" + col]).mean())
        for pool in ("all_named", "alternatives"):
            ex_t = row["targeted_adj"] - 1
            ex_r = row["{}_adj_mean".format(pool)] - 1
            ok = np.isfinite(ex_r) and ex_r > 0
            row["TR_excess_{}".format(pool)] = ex_t / ex_r if ok else np.nan
            ra = row["{}_adj_mean".format(pool)]
            row["TR_ratio_{}".format(pool)] = (row["targeted_adj"] / ra
                                               if np.isfinite(ra) and ra > 0
                                               else np.nan)
        g.append(row)
    pd.DataFrame(g).to_csv(os.path.join(out, "asym_summary.csv"),
                           index=False)
    print("  asym summary written")


# ---------------------------------------------------------------------------
def do_validate(city, out, cg):
    vdir = os.path.join(BASE, RESULTS_DIR, city, "asym")
    sim, pair_ids = load_sim(vdir)
    pairs = pd.read_csv(os.path.join(vdir, "pairs.csv"))
    sets_of = pairs.set_index("pair").loc[pair_ids, "set"].values
    wts = pair_weights(vdir, pair_ids)
    rk = pd.read_csv(os.path.join(BASE, OUTPUTS_DIR, city,
                                  "rankings_for_validation.csv"))
    rows, boot_rows, store = [], [], {}
    t0 = time.time()
    for name in RANKINGS_VAL:
        order = rk.sort_values([name, "road_name"])["road_name"].tolist()
        for k in K_VAL:
            mask, _, _ = cl.roads_closure(cg, order[:k])
            res, _ = sim.run(mask, frag=False)
            for m in LAYERS:
                for sets in (("B",), ("A",), ("A", "B")):
                    rt = pair_rt(sim, pair_ids, sets_of, res, m, sets,
                                 weights=wts)
                    store[(name, k, m, "".join(sets))] = rt
                    rows.append(dict(ranking=name, k=k, layer=m,
                                     sets="".join(sets), **summarize(rt)))
        print("   validated {} ({:.0f} s)".format(name, time.time() - t0))
    # random selections: the alternatives pool of the asym runs
    var = Variant(vdir)
    met = scenario_metrics(var, sets=("B",), rule="iso")
    met = met.merge(var.scen[["scenario", "kind", "k", "pool"]],
                    on="scenario")
    rnd = met[(met.kind == "random") & (met.pool == "alternatives")
              & met.k.isin(K_VAL)]
    for (k, m), d in rnd.groupby(["k", "layer"]):
        rows.append(dict(ranking="Random_mean", k=k, layer=m, sets="B",
                         n_valid=int(d.n_valid.mean()), L=d.L.mean(),
                         e=d.e.mean(), adj=d.adj.mean()))
        rows.append(dict(ranking="Random_sd", k=k, layer=m, sets="B",
                         L=d.L.std(ddof=1), e=d.e.std(ddof=1),
                         adj=d.adj.std(ddof=1)))
        rows.append(dict(ranking="Random_max", k=k, layer=m, sets="B",
                         L=d.L.max(), e=d.e.max(), adj=d.adj.max()))
    val = pd.DataFrame(rows)
    val.to_csv(os.path.join(out, "validation_topk.csv"), index=False)
    # share of random selections at least as harmful (set B)
    ge = []
    for name in RANKINGS_VAL:
        for k in K_VAL:
            for m in LAYERS:
                v = val[(val.ranking == name) & (val.k == k)
                        & (val.layer == m) & (val.sets == "B")]["adj"]
                r = rnd[(rnd.k == k) & (rnd.layer == m)]["adj"]
                ge.append(dict(ranking=name, k=k, layer=m,
                               random_share_ge=float((r >= v.iloc[0]).mean()),
                               n_random=int(len(r))))
    pd.DataFrame(ge).to_csv(os.path.join(out, "validation_vs_random.csv"),
                            index=False)
    # paired bootstrap over OD pairs (set B, low-stress layer)
    rng = np.random.default_rng(11)
    for other in ("CarOnly_MCDM", "PageRank_only", "IPS_original",
                  "NR_TOPSIS"):
        for k in K_VAL:
            for m in ("bike_safe", "car"):
                a = store[("Consensus", k, m, "B")]
                b = store[(other, k, m, "B")]
                common = a.index.intersection(b.index)
                da = a.loc[common, "rt"].values
                db = b.loc[common, "rt"].values
                ww = a.loc[common, "w"].values
                diff = da - db
                la = (da == P).astype(float) - (db == P).astype(float)
                n = len(diff)
                bs, bl = np.empty(N_BOOT), np.empty(N_BOOT)
                for t in range(N_BOOT):
                    ix = rng.integers(0, n, n)
                    wsum = ww[ix].sum()
                    bs[t] = (ww[ix] * diff[ix]).sum() / wsum
                    bl[t] = (ww[ix] * la[ix]).sum() / wsum
                diff_mean = float((ww * diff).sum() / ww.sum())
                la_mean = float((ww * la).sum() / ww.sum())
                boot_rows.append(dict(
                    comparison="Consensus_minus_" + other, k=k, layer=m,
                    n_pairs=int(len(common)),
                    diff_adj=diff_mean,
                    ci_low=float(np.percentile(bs, 2.5)),
                    ci_high=float(np.percentile(bs, 97.5)),
                    diff_L=la_mean,
                    ciL_low=float(np.percentile(bl, 2.5)),
                    ciL_high=float(np.percentile(bl, 97.5))))
    pd.DataFrame(boot_rows).to_csv(os.path.join(out, "validation_bootstrap.csv"),
                                   index=False)
    print("  validation written")


# ---------------------------------------------------------------------------
def do_invest(city, out, cg):
    vdir = os.path.join(BASE, RESULTS_DIR, city, "asym")
    sim, pair_ids = load_sim(vdir)
    core = sim.core
    o, d = sim.access["bike_safe"]
    base = sim.dist0["bike_safe"]
    wts = pair_weights(vdir, pair_ids)
    rk = pd.read_csv(os.path.join(BASE, OUTPUTS_DIR, city,
                                  "rankings_for_validation.csv"))
    alts = rk["road_name"].tolist()
    rng = np.random.default_rng(5)

    def gain(roads):
        E = cg.edges_of_roads(roads)
        extra = np.zeros(core.m, dtype=bool)
        extra[E] = True
        lts2 = core.lts.copy()
        lts2[E] = 1.0
        lay = Layer(core, "bike_safe", "length", extra_mask=extra,
                    lts_override=lts2)
        _, GT = lay.graphs(None)
        # origins/destinations keep their access junctions
        new, _ = route_pairs(GT, o, d)
        was = np.isfinite(base)
        now = np.isfinite(new)
        both = was & now & (base > 0)
        return {"newly_reachable": int((now & ~was).sum()),
                "reach_gain_pop_share": float(wts[now & ~was].sum()
                                              / wts.sum()),
                "reach_share_before": float(was.mean()),
                "reach_share_after": float(now.mean()),
                "mean_rel_dist_change": float(((new[both] - base[both])
                                               / base[both]).mean()),
                "upgraded_km_directed": float(core.length[E].sum() / 1000)}

    rows = []
    for name in RANKINGS_VAL:
        order = rk.sort_values([name, "road_name"])["road_name"].tolist()
        for k in K_INV:
            rows.append(dict(ranking=name, k=k, rep=0, **gain(order[:k])))
    for k in K_INV:
        for rep in range(20):
            pick = list(rng.choice(alts, size=k, replace=False))
            rows.append(dict(ranking="Random", k=k, rep=rep, **gain(pick)))
    pd.DataFrame(rows).to_csv(os.path.join(out, "investment_gain.csv"),
                              index=False)
    print("  investment written")


# ---------------------------------------------------------------------------
def do_stats(city, out, cg):
    exp = os.path.join(BASE, "graph_exports", city)
    res = os.path.join(BASE, RESULTS_DIR, city)
    with open(os.path.join(res, "prepare.json"), encoding="utf-8") as f:
        prep = json.load(f)
    seg = cg.seg
    seglen = pd.Series(cg.length).groupby(seg).min()

    def km(mask):
        s = np.unique(seg[mask])
        return float(seglen.loc[s].sum() / 1000)
    lay_mask = {"car": cg.driveable, "bike_all": cg.bike_allowed,
                "bike_safe": cg.bike_allowed & (cg.lts <= 2)}
    cls1 = np.array([1 if h == "cycleway" else 0 for h in cg.highway]) > 0
    from mmsim import lts as L
    bclass = np.array([L.bicycle_class(p, L.CLASS2_MAIN) if b else 0
                       for p, b in zip(cg.props, cg.bike_allowed)])
    st = {
        "directed_edges": int(cg.m), "physical_segments": int(cg.n_seg),
        "junctions": int(cg.n),
        "mean_physical_segment_m": float(seglen.mean()),
        "driveable_km": km(lay_mask["car"]),
        "cycling_km": km(lay_mask["bike_all"]),
        "low_stress_km": km(lay_mask["bike_safe"]),
        "cycleway_km": km(cls1),
        "protected_class1_km": km(cg.bike_allowed & (bclass == 1)),
        "driveable_km_directed_sum": float(cg.length[cg.driveable].sum()
                                           / 1000),
        "low_stress_km_directed_sum": float(
            cg.length[lay_mask["bike_safe"]].sum() / 1000),
        "tagged_speed_share_driveable_segments": float(pd.Series(
            cg.speed_tagged[cg.driveable]).groupby(
            seg[cg.driveable]).max().mean()),
        "residents_in_area": prep["residents_in_area"],
        "residents_municipality": prep["residents_municipality"],
        "census_sections_in_area": prep["census_sections_in_area"],
        "named_roads": len(cg.road_pagerank),
        "alternatives": prep["alternatives"],
        "pois_usable": prep["pois_usable"],
        "pois_polygons": prep["pois_polygons"],
    }
    # LTS variants: how many edges change low-stress status
    for var in ("stored", "zone30"):
        c2 = CityGraph(exp, var, verbose=False)
        a = np.nan_to_num(c2.lts, nan=9) <= 2
        b = np.nan_to_num(cg.lts, nan=9) <= 2
        st["lowstress_status_changed_vs_{}_km".format(var)] = km(a != b)
        st["lowstress_status_changed_vs_{}_edges".format(var)] = int(
            (a != b).sum())
    orig = np.array([np.nan if (x := L.edge_lts(p, "orig")) is None else x
                     for p in cg.props], dtype=float)
    same = (np.nan_to_num(orig, nan=-1) == np.nan_to_num(cg.lts_stored,
                                                          nan=-1))
    st["lts_recomputed_equals_stored_share"] = float(same.mean())
    minor = np.isin(cg.highway, ["residential", "unclassified",
                                 "living_street"])
    st["bus_minor_imputed_speed_km"] = km(minor & cg.on_bus
                                          & ~cg.speed_tagged)
    st["minor_imputed_speed_share"] = float((~cg.speed_tagged[minor]).mean())
    with open(os.path.join(out, "network_stats.json"), "w") as f:
        json.dump(st, f, indent=2)

    # share of each layer on the alternatives (top-half roads)
    alts = pd.read_csv(os.path.join(res, "alternatives.csv"))["road_name"]
    on_alt = np.zeros(cg.m, dtype=bool)
    on_alt[cg.edges_of_roads(alts.tolist())] = True
    sh = {m: km(lay_mask[m] & on_alt) / max(km(lay_mask[m]), 1e-9)
          for m in LAYERS}
    # route overlap (edge-based) between low-stress and car baseline paths
    sim, pair_ids = load_sim(os.path.join(res, "asym"))
    ov = []
    for i in range(len(pair_ids)):
        a = sim.pk["bike_safe"][sim.pp["bike_safe"] == i]
        b = sim.pk["car"][sim.pp["car"] == i]
        if len(a) and len(b):
            nn = cg.n
            ua = {(min(x // nn, x % nn), max(x // nn, x % nn)) for x in a}
            ub = {(min(x // nn, x % nn), max(x // nn, x % nn)) for x in b}
            ov.append(len(ua & ub) / len(ua))
    struct = {"share_on_alternatives_" + m: sh[m] for m in LAYERS}
    struct["edge_overlap_lowstress_car_mean"] = float(np.mean(ov))
    struct["edge_overlap_n_pairs"] = len(ov)
    for m in LAYERS:
        struct["baseline_reachable_" + m] = int(
            np.isfinite(sim.dist0[m]).sum())
    struct["n_pairs"] = int(len(pair_ids))
    struct["coverage_lowstress"] = (struct["baseline_reachable_bike_safe"]
                                    / struct["baseline_reachable_car"])
    lay = {m: Layer(sim.core, m) for m in LAYERS}
    struct["lcc_lowstress_over_car_nodes"] = float(
        lay["bike_safe"].lcc0.sum() / lay["car"].lcc0.sum())
    with open(os.path.join(out, "structure.json"), "w") as f:
        json.dump(struct, f, indent=2)

    # distance of every alternative to the study-area edge
    s, w, n_, e = cg.bbox
    corner = cg.proj([s, n_], [w, e])
    rows = []
    for r in alts:
        E = cg.road_edges[r]
        nodes = np.unique(np.concatenate([cg.u[E], cg.v[E]]))
        xy = cg.xy[nodes]
        dist = np.min(np.minimum.reduce([xy[:, 0] - corner[0, 0],
                                         corner[1, 0] - xy[:, 0],
                                         xy[:, 1] - corner[0, 1],
                                         corner[1, 1] - xy[:, 1]]))
        rows.append({"road_name": r, "dist_to_bbox_edge_m": float(dist)})
    pd.DataFrame(rows).to_csv(os.path.join(out, "road_boundary.csv"),
                              index=False)
    print("  stats written")


# ---------------------------------------------------------------------------
BYPASS_FACTOR = 1.3


def do_bypass(city, out, cg, n_random_roads=200, k_list=(10, 20),
              n_sets=50):
    """
    Redundancy experiment: every selected road receives a
    parallel low-stress bypass (one LTS-1 edge per physical segment, both
    directions, BYPASS_FACTOR times longer, not closed with the road). The
    road is then closed as in the single-road experiment and the low-stress
    harm H = adjusted detour - 1 is measured on the held-out set B with and
    without the bypass. The avoided harm of a road is the value of a
    parallel route there; a ranking is evaluated by the avoided harm of its
    top-k roads, against random sets of k alternatives.
    """
    vdir = os.path.join(BASE, RESULTS_DIR, city, "single_main")
    sim, pair_ids = load_sim(vdir)
    pairs = pd.read_csv(os.path.join(vdir, "pairs.csv"))
    sets_of = pairs.set_index("pair").loc[pair_ids, "set"].values
    wts = pair_weights(vdir, pair_ids)
    core = sim.core
    lay0 = sim.layers["bike_safe"]
    o, d = sim.access["bike_safe"]
    base = sim.dist0["bike_safe"]
    inB = (sets_of == "B") & np.isfinite(base) & (base > 0)
    rk = pd.read_csv(os.path.join(BASE, OUTPUTS_DIR, city,
                                  "rankings_for_validation.csv"))
    alts = rk["road_name"].tolist()
    rng = np.random.default_rng(13)
    sel = set()
    rank_cols = [c for c in RANKINGS_VAL if c in rk.columns]
    for name in rank_cols:
        order = rk.sort_values([name, "road_name"])["road_name"].tolist()
        sel.update(order[:max(k_list)])
    rand_pool = list(rng.choice(alts, size=min(n_random_roads, len(alts)),
                                replace=False))
    sel.update(rand_pool)

    def harm(new, excl):
        valid = inB & ~excl
        with np.errstate(invalid="ignore", divide="ignore"):
            rt = np.where(np.isfinite(new),
                          new / np.where(base > 0, base, 1), P)
        rt = np.where(valid, rt, 0.0)
        w = wts * valid
        return float((w * rt).sum() / w.sum()) - 1.0

    rows = []
    t0 = time.time()
    for r in sorted(sel):
        closed = cl.single_closure(cg, r, 0.25)[0]
        E = cg.road_edges[r]
        segs = np.unique(cg.seg[E])
        # one bypass edge pair per physical segment of the road
        first_e = {}
        for e in E:
            first_e.setdefault(cg.seg[e], e)
        eu = np.array([cg.u[first_e[sg]] for sg in segs])
        ev = np.array([cg.v[first_e[sg]] for sg in segs])
        el = np.array([pd.Series(cg.length[cg.seg == sg]).min()
                       for sg in segs]) * BYPASS_FACTOR
        nb = len(segs)
        ext = Core(core.n,
                   np.concatenate([core.u, eu, ev]),
                   np.concatenate([core.v, ev, eu]),
                   np.concatenate([core.length, el, el]),
                   np.concatenate([core.driveable, np.zeros(2 * nb, bool)]),
                   np.concatenate([core.bike_allowed, np.ones(2 * nb, bool)]),
                   np.concatenate([core.lts, np.ones(2 * nb)]),
                   core.pop)
        closed_ext = np.concatenate([closed, np.zeros(2 * nb, bool)])
        # affected pairs and excluded endpoints (no-bypass closure)
        ck = lay0.closed_keys(closed)
        hit = np.isin(sim.pk["bike_safe"], ck)
        aff = np.unique(sim.pp["bike_safe"][hit])
        aff = aff[inB[aff]]
        deg = lay0.degree_after(closed)
        iso_nodes = lay0.nodes & (deg == 0)
        excl = iso_nodes[o] | iso_nodes[d]
        _, GT0 = lay0.graphs(closed)
        new0 = base.copy()
        if len(aff):
            new0[aff], _ = route_pairs(GT0, o[aff], d[aff])
        lay1 = Layer(ext, "bike_safe")
        _, GT1 = lay1.graphs(closed_ext)
        new1 = base.copy()
        if len(aff):
            new1[aff], _ = route_pairs(GT1, o[aff], d[aff])
        h0, h1 = harm(new0, excl), harm(new1, excl)
        rows.append({"road_name": r, "harm_no_bypass": h0,
                     "harm_with_bypass": h1, "avoided": h0 - h1,
                     "bypass_km": float(el.sum() / 1000),
                     "in_random_pool": r in rand_pool})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "bypass_per_road.csv"), index=False)
    av = df.set_index("road_name")["avoided"]
    res = []
    for name in rank_cols:
        order = rk.sort_values([name, "road_name"])["road_name"].tolist()
        for k in k_list:
            res.append({"ranking": name, "k": k,
                        "avoided_sum": float(av.loc[order[:k]].sum())})
    pool = df[df.in_random_pool]["avoided"].values
    for k in k_list:
        sums = np.array([rng.choice(pool, size=k, replace=False).sum()
                         for _ in range(n_sets)])
        res.append({"ranking": "Random_mean", "k": k,
                    "avoided_sum": float(sums.mean())})
        res.append({"ranking": "Random_p95", "k": k,
                    "avoided_sum": float(np.percentile(sums, 95))})
    pd.DataFrame(res).to_csv(os.path.join(out, "bypass_summary.csv"),
                             index=False)
    print("  bypass written ({:.0f} s, {} roads)".format(time.time() - t0,
                                                         len(sel)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--what", default="stats,asym,validate,invest,bypass")
    a = ap.parse_args()
    out = os.path.join(BASE, OUTPUTS_DIR, a.city)
    os.makedirs(out, exist_ok=True)
    what = a.what.split(",")
    cg = None
    if {"stats", "validate", "invest", "bypass"} & set(what):
        cg = CityGraph(os.path.join(BASE, "graph_exports", a.city), "main",
                       verbose=False)
    if "stats" in what:
        do_stats(a.city, out, cg)
    if "asym" in what:
        do_asym(a.city, out)
    if "validate" in what:
        do_validate(a.city, out, cg)
    if "invest" in what:
        do_invest(a.city, out, cg)
    if "bypass" in what:
        do_bypass(a.city, out, cg)


if __name__ == "__main__":
    main()
