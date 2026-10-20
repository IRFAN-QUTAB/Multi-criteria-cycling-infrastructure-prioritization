import argparse
import json
import os
import random
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from vulnerability_analysis import VulnerabilityAnalyzer  # noqa: E402
from mcdm_analysis import CITY_FILES, load_csv  # noqa: E402

RANK_COLUMNS = {
    "Consensus": "consensus_rank",
    "NR_TOPSIS": "NR_TOPSIS_rank",
    "CarOnly_MCDM": "CarOnly_MCDM_rank",
    "IPS_original": "IPS_original_rank",
}
MODES = ["car", "bike_safe", "bike_all"]
METRICS = ["avg_detour_ratio_adj", "unreachable_pct", "pois_lost", "wcc_delta"]


def run_files(city):
    csv = os.path.join(ROOT, CITY_FILES[city])
    base = csv[:-4]
    return {"per_road": csv, "run_json": base + ".json",
            "od_json": base + "_od_pairs.json",
            "simultaneous": base + "_simultaneous.csv"}


def check_od_nodes(analyzer, od_pairs, tol=1e-6):
    """Every origin id must exist at the saved coordinates."""
    ids = [int(o["origin_id"]) for o in od_pairs]
    rows = analyzer._run("""
        MATCH (n) WHERE id(n) IN $ids
        RETURN id(n) AS id, n.lat AS lat, n.lon AS lon""", ids=ids)
    found = {r["id"]: (r["lat"], r["lon"]) for r in rows}
    bad = []
    for o in od_pairs:
        got = found.get(int(o["origin_id"]))
        if got is None or got[0] is None or \
                abs(got[0] - float(o["origin_lat"])) > tol or \
                abs(got[1] - float(o["origin_lon"])) > tol:
            bad.append(o["origin_id"])
    dest_ids = list({int(o["dest_junction_id"]) for o in od_pairs})
    n_dest = analyzer._run("MATCH (n) WHERE id(n) IN $ids RETURN count(n) AS c",
                           ids=dest_ids)[0]["c"]
    return bad, len(dest_ids) - n_dest


def measure(analyzer, road_names):
    analyzer.reset_all()
    analyzer._disrupt_multiple_roads(list(road_names))
    try:
        m = analyzer._measure_modes_quick()
    finally:
        analyzer.reset_all()
    row = {}
    for mode in MODES:
        for key in METRICS:
            row["{}_{}".format(mode, key)] = m[mode][key]
    row["MID_adj"] = (m["bike_safe"]["avg_detour_ratio_adj"]
                      - m["car"]["avg_detour_ratio_adj"])
    return row


def main():
    ap = argparse.ArgumentParser(description="Top-k closure validation")
    ap.add_argument("--neo4jURL", "-n", required=True)
    ap.add_argument("--neo4juser", "-u", required=True)
    ap.add_argument("--neo4jpwd", "-p", required=True)
    ap.add_argument("--city", "-c", required=True, choices=list(CITY_FILES))
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 30, 50])
    ap.add_argument("--random-iter", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primary-weights", default="critic")
    ap.add_argument("--mcdm-out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--force", action="store_true",
                    help="continue even if the baseline differs from the paper run")
    args = ap.parse_args()

    city = args.city
    f = run_files(city)
    for key, path in f.items():
        if not os.path.exists(path):
            sys.exit("Missing paper-run file ({}): {}".format(key, path))
    rank_file = os.path.join(args.mcdm_out, city,
                             "scores_rankings_{}.csv".format(args.primary_weights))
    if not os.path.exists(rank_file):
        sys.exit("Run mcdm_analysis.py first: {} not found".format(rank_file))

    run = json.load(open(f["run_json"], encoding="utf-8"))
    od_pairs = json.load(open(f["od_json"], encoding="utf-8"))
    for o in od_pairs:          # destinations are already network-resolved
        o["dest_candidates"] = []
    sc = pd.read_csv(rank_file)
    per_road = load_csv(f["per_road"]).sort_values("rank", kind="mergesort")
    pagerank_order = per_road["road_name"].tolist()      # paper order
    pool = sc["road_name"].tolist()

    print("Paper run: {} | {} OD pairs | weight={} | penalty={} | "
          "population-weighted={}".format(
              os.path.basename(f["run_json"]), len(od_pairs), run.get("weight"),
              run.get("unreachable_penalty"), run.get("population_weighted")))

    analyzer = VulnerabilityAnalyzer(
        args.neo4jURL, args.neo4juser, args.neo4jpwd,
        closure_pct=run.get("closure_pct", 0.25),
        closure_min=run.get("closure_min", 10),
        weight=run.get("weight", "distance"),
        unreachable_penalty=run.get("unreachable_penalty", 5.0))
    rows = []
    t0 = time.time()
    try:
        # ---- (a) the loaded graph must be the paper's graph ----
        bad, missing_dest = check_od_nodes(analyzer, od_pairs)
        if bad or missing_dest:
            sys.exit("Graph mismatch: {} origins not found at saved coordinates, "
                     "{} destination nodes missing. Is the {} graph loaded?"
                     .format(len(bad), missing_dest, city))
        print("OD nodes found in the graph at the saved coordinates: OK")

        # ---- (b) baseline must equal the paper baseline ----
        analyzer.od_pairs = od_pairs
        analyzer.compute_baseline()
        saved = run.get("baseline", {})
        now = {m: sum(1 for d in analyzer.baseline[m] if d is not None) for m in MODES}
        mism = {m: (now[m], saved.get("{}_reachable".format(m)))
                for m in MODES if now[m] != saved.get("{}_reachable".format(m))}
        print("Baseline reachable now / paper: " + ", ".join(
            "{} {}/{}".format(m, now[m], saved.get("{}_reachable".format(m)))
            for m in MODES))
        if mism and not args.force:
            sys.exit("Baseline differs from the paper run {} - stopped "
                     "(use --force to continue anyway).".format(mism))

        # ---- closures ----
        rankings = {name: sc.sort_values(col, kind="mergesort")["road_name"].tolist()
                    for name, col in RANK_COLUMNS.items()}
        rankings["PageRank_only"] = pagerank_order
        for k in args.k:
            for name, order in rankings.items():
                row = {"k": k, "ranking": name, "iteration": 0}
                row.update(measure(analyzer, order[:k]))
                rows.append(row)
                print("k={:3d} {:14s} bike_safe={:.3f} car={:.3f} MID={:+.3f}".format(
                    k, name, row["bike_safe_avg_detour_ratio_adj"],
                    row["car_avg_detour_ratio_adj"], row["MID_adj"]))
            rng = random.Random(args.seed + k)
            for it in range(args.random_iter):
                row = {"k": k, "ranking": "Random", "iteration": it}
                row.update(measure(analyzer, rng.sample(pool, k)))
                rows.append(row)
            print("  ({:.0f}s elapsed)".format(time.time() - t0))
    finally:
        analyzer.reset_all()
        analyzer.close()

    out_dir = os.path.join(args.mcdm_out, city)
    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(out_dir, "validation_topk_closures.csv"), index=False)

    value_cols = [c for c in detail.columns if c not in ("k", "ranking", "iteration")]
    g = detail.groupby(["k", "ranking"])[value_cols]
    summ = g.mean().add_suffix("_mean").join(g.std().fillna(0.0).add_suffix("_sd"))
    summ = summ.reset_index()
    rnd = summ[summ.ranking == "Random"].set_index("k")
    summ["bike_safe_detour_excess_vs_random"] = [
        (r["bike_safe_avg_detour_ratio_adj_mean"] - 1.0)
        / max(rnd.loc[r["k"], "bike_safe_avg_detour_ratio_adj_mean"] - 1.0, 1e-9)
        for _, r in summ.iterrows()]
    summ.to_csv(os.path.join(out_dir, "validation_topk_summary.csv"), index=False)

    # ---- reproducibility: PageRank closures vs paper simultaneous results ----
    paper = load_csv(f["simultaneous"]).set_index("k")
    rep = []
    pr = detail[detail.ranking == "PageRank_only"].set_index("k")
    for k in pr.index:
        if k not in paper.index:
            continue
        for mode in MODES:
            a = pr.loc[k, "{}_avg_detour_ratio_adj".format(mode)]
            b = paper.loc[k, "{}_avg_detour_adj".format(mode)]
            rep.append({"k": k, "mode": mode, "this_run": a, "paper": b,
                        "abs_diff": abs(a - b)})
    rep = pd.DataFrame(rep)
    rep.to_csv(os.path.join(out_dir, "validation_reproducibility.csv"), index=False)
    if len(rep):
        print("\nReproducibility (PageRank top-k vs paper), max abs diff = {:.4f}"
              .format(rep.abs_diff.max()))
        print(rep.round(4).to_string(index=False))
    print("\nSaved to", out_dir)


if __name__ == "__main__":
    main()
