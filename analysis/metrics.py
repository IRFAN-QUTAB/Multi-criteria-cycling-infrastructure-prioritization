"""
Accessibility metrics from the raw scenario outputs of run_city.py.

For a scenario s, a layer m and a set of OD pairs S (A, B or AB):

  R0      pairs of S reachable at baseline in m, with a positive cost
  X       pairs of R0 whose origin or destination junction is cut out by
          the closure itself (rule "iso": every edge of the junction in m is
          closed; rule "inc": the junction touches a closed edge; rule
          "none": nothing excluded = Neo4j version). These trips start or end
          inside the works and are reported separately.
  V       R0 minus X (valid pairs)
  lost    pairs of V that become unreachable
  L       |lost| / |V|                         service loss share
  e       mean of (r - 1) over V minus lost    conditional excess detour
  adj     mean over V of r~ (r~ = P if lost)   adjusted detour, baseline-
                                               reachable pairs only
  legacy  earlier definition of the adjusted detour: mean over all pairs of S, 1 for pairs
          unreachable at baseline, P for lost, no exclusion
  cut     population share of the layer's main component that is
          disconnected from it, closure junctions excluded (C4)
"""

import glob
import json
import os

import numpy as np
import pandas as pd

LAYERS = ("car", "bike_all", "bike_safe")
P_DEFAULT = 5.0
RESULTS_DIR = os.environ.get("MMSIM_RESULTS", "results")
OUTPUTS_DIR = os.environ.get("MMSIM_OUTPUTS", "outputs")


class Variant:
    def __init__(self, vdir):
        self.vdir = vdir
        with open(os.path.join(vdir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.pairs = pd.read_csv(os.path.join(vdir, "pairs.csv"))
        self.scen = pd.read_csv(os.path.join(vdir, "scenarios.csv"))
        parts = sorted(glob.glob(os.path.join(vdir, "changes_part*.csv.gz")))
        self.changes = (pd.concat([pd.read_csv(p) for p in parts],
                                  ignore_index=True)
                        if parts else pd.DataFrame(
            columns=["scenario", "layer", "pair", "new_dist", "iso", "inc"]))
        self.changes["new_dist"] = self.changes["new_dist"].astype(float)
        self.changes = self.changes.drop_duplicates(
            ["scenario", "layer", "pair"], keep="last")
        fparts = sorted(glob.glob(os.path.join(vdir, "frag_part*.csv")))
        self.frag = (pd.concat([pd.read_csv(p) for p in fparts],
                               ignore_index=True).drop_duplicates(
            ["scenario", "layer"], keep="last") if fparts else None)
        with open(os.path.join(vdir, "done.txt")) as f:
            self.done = {int(x) for x in f.read().split()}
        if "weight" in self.pairs.columns:
            self.weight = pd.Series(self.pairs["weight"].astype(float).values,
                                    index=self.pairs["pair"].values)
        else:
            self.weight = pd.Series(1.0, index=self.pairs["pair"].values)
        self.base = {}
        for m in LAYERS:
            b = self.pairs["dist0_" + m].astype(float).values
            self.base[m] = pd.Series(b, index=self.pairs["pair"].values)


def scenario_metrics(var, sets=("A", "B"), rule="iso", P=P_DEFAULT,
                     scenarios=None, pair_ids=None):
    """
    One row per (scenario, layer). Every OD pair carries the population
    weight of its origin (column 'weight' of pairs.csv; 1 if absent), so all
    shares and means are population-weighted. pair_ids overrides sets.
    """
    if pair_ids is not None:
        pid = np.asarray(sorted(set(pair_ids)
                                & set(var.pairs["pair"].values)))
    else:
        pairs = var.pairs[var.pairs["set"].isin(sets)]
        pid = pairs["pair"].values
    wser = var.weight
    ch = var.changes[var.changes["pair"].isin(pid)]
    if scenarios is not None:
        ch = ch[ch["scenario"].isin(scenarios)]
    scen_ids = sorted(var.done if scenarios is None
                      else set(scenarios) & var.done)
    out = []
    for m in LAYERS:
        b = var.base[m].loc[pid].values
        w = wser.loc[pid].values
        r0 = np.isfinite(b) & (b > 0)
        W_r0 = float(w[r0].sum())
        W_set = float(w.sum())
        base_of = pd.Series(b, index=pid)
        cm = ch[ch["layer"] == m].copy()
        cm["base"] = base_of.loc[cm["pair"].values].values
        cm["w"] = wser.loc[cm["pair"].values].values
        cm = cm[np.isfinite(cm["base"]) & (cm["base"] > 0)]
        if rule == "iso":
            cm["excl"] = cm["iso"].astype(bool)
        elif rule == "inc":
            cm["excl"] = cm["inc"].astype(bool)
        else:
            cm["excl"] = False
        cm["lost"] = ~np.isfinite(cm["new_dist"])
        r = np.where(cm["lost"], np.nan, cm["new_dist"] / cm["base"])
        cm["r"] = r
        cm["rt"] = np.where(cm["lost"], P, r)
        valid = ~cm["excl"]
        cm["w_excl"] = cm["w"] * cm["excl"]
        cm["w_lost"] = cm["w"] * (cm["lost"] & valid)
        cm["w_ex"] = cm["w"] * np.where(~cm["lost"] & valid,
                                        cm["r"].fillna(1.0) - 1.0, 0.0)
        cm["w_rt_v"] = cm["w"] * np.where(valid, cm["rt"], 0.0)
        cm["w_v"] = cm["w"] * valid
        cm["w_lost_all"] = cm["w"] * cm["lost"]
        cm["w_rt_all"] = cm["w"] * cm["rt"]
        cm["n_excl"] = cm["excl"].astype(int)
        cm["n_lost"] = (cm["lost"] & valid).astype(int)
        agg = cm.groupby("scenario")[["w_excl", "w_lost", "w_ex", "w_rt_v",
                                      "w_v", "w_lost_all", "w_rt_all", "w",
                                      "n_excl", "n_lost"]].sum()
        agg = agg.reindex(scen_ids).fillna(0.0)
        W_valid = W_r0 - agg["w_excl"]
        W_surv = (W_valid - agg["w_lost"]).clip(lower=1e-12)
        res = pd.DataFrame({
            "scenario": scen_ids, "layer": m,
            "n_set": len(pid), "n_r0": int(r0.sum()),
            "n_excl": agg["n_excl"].values.astype(int),
            "n_valid": (int(r0.sum()) - agg["n_excl"]).values.astype(int),
            "n_lost": agg["n_lost"].values.astype(int),
            "w_valid": W_valid.values,
        })
        Wv = W_valid.clip(lower=1e-12)
        res["L"] = (agg["w_lost"] / Wv).values
        res["e"] = (agg["w_ex"] / W_surv).values
        res["adj"] = ((W_valid - agg["w_v"] + agg["w_rt_v"]) / Wv).values
        res["legacy_adj"] = ((W_set - agg["w"] + agg["w_rt_all"])
                             / W_set).values
        res["legacy_lost"] = (agg["w_lost_all"] / W_set).values
        empty = res["n_valid"].values <= 0
        res.loc[empty, ["L", "e", "adj"]] = np.nan
        out.append(res)
    df = pd.concat(out, ignore_index=True)
    if var.frag is not None:
        fr = var.frag.copy()
        col = {"iso": "cut_xiso_pop", "inc": "cut_xinc_pop",
               "none": "cut_pop"}[rule]
        coln = {"iso": "cut_xiso_n", "inc": "cut_xinc_n",
                "none": "cut_n"}[rule]
        fr["cut"] = fr[col] / fr["lcc0_pop"].replace(0, np.nan)
        fr["cut_nodes_share"] = fr[coln] / fr["lcc0_n"].replace(0, np.nan)
        df = df.merge(fr[["scenario", "layer", "cut", "cut_nodes_share",
                          "rerouted", "iso_n"]],
                      on=["scenario", "layer"], how="left")
    return df


def wide(df, cols=("L", "e", "adj", "legacy_adj", "cut", "n_valid",
                   "n_excl", "n_lost")):
    w = df.pivot(index="scenario", columns="layer",
                 values=[c for c in cols if c in df.columns])
    w.columns = ["{}_{}".format(a, b) for a, b in w.columns]
    return w.reset_index()


def affected(L, e, tol=0.01):
    return (L > 0) | (e > tol)


def compensation_class(Lc, ec, Lb, eb, tol=0.01):
    car = affected(Lc, ec, tol)
    bik = affected(Lb, eb, tol)
    out = np.where(
        car,
        np.where(~bik, "full",
                 np.where((Lb <= Lc) & (eb <= ec) & ((Lb < Lc) | (eb < ec)),
                          "partial", "none")),
        np.where(bik, "bike_only_vulnerable", "no_impact"))
    return out


def pair_level(var, scenario, layer, sets=("A", "B"), rule="iso",
               P=P_DEFAULT):
    """Per-pair r~ for one scenario and layer (valid pairs only)."""
    pairs = var.pairs[var.pairs["set"].isin(sets)]
    pid = pairs["pair"].values
    b = var.base[layer].loc[pid].values
    ok = np.isfinite(b) & (b > 0)
    rt = pd.Series(np.ones(len(pid)), index=pid)
    ch = var.changes[(var.changes["scenario"] == scenario)
                     & (var.changes["layer"] == layer)
                     & var.changes["pair"].isin(pid)]
    excl = set()
    for _, row in ch.iterrows():
        p = int(row["pair"])
        bb = var.base[layer].loc[p]
        if not (np.isfinite(bb) and bb > 0):
            continue
        if (rule == "iso" and row["iso"]) or (rule == "inc" and row["inc"]):
            excl.add(p)
            continue
        nd = row["new_dist"]
        rt.loc[p] = P if not np.isfinite(nd) else nd / bb
    keep = [p for p, o in zip(pid, ok) if o and p not in excl]
    return rt.loc[keep]
