

import argparse
import csv
import datetime as _dt
import gzip
import json
import multiprocessing as mp
import os
import pickle
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = HERE

try:
    import numpy as np  # noqa: E402
    import scipy  # noqa: F401,E402
except ImportError:
    raise SystemExit("numpy/scipy missing in this Python environment: "
                     "pip install numpy scipy")

from mmsim import closures as cl  # noqa: E402
from mmsim import od as odm  # noqa: E402
from mmsim.citygraph import CityGraph  # noqa: E402
from mmsim.network import LAYERS, Core, Layer, Simulator  # noqa: E402
from mmsim import parallel as par  # noqa: E402

OLD_RUNS = {
    "modena": "modena_results/modena_20260711_005543_pct25",
    "ferrara": "ferrara/ferrara_20260711_075849_pct25",
    "pesaro": "pesaro/pesaro_20260711_084658_pct25",
}
VARIANTS = [
    # name,          lts,      cost,     access,       dest,   depth, pairs
    ("main",         "main",   "length", "nearest100", "car",   0.25, "AB"),
    ("depth10",      "main",   "length", "nearest100", "car",   0.10, "A"),
    ("depth50",      "main",   "length", "nearest100", "car",   0.50, "A"),
    ("depth100",     "main",   "length", "nearest100", "car",   1.00, "A"),
    ("cost_lts",     "main",   "lts",    "nearest100", "car",   0.25, "A"),
    ("lts_stored",   "stored", "length", "nearest100", "car",   0.25, "A"),
    ("lts_zone30",   "zone30", "length", "nearest100", "car",   0.25, "A"),
    ("access_legacy", "main",  "length", "legacy",     "car",   0.25, "A"),
    ("dest_layer",   "main",   "length", "nearest100", "layer", 0.25, "A"),
]
K_ASYM = (5, 10, 20, 30, 50, 70, 100)
RESULTS_DIR = "results"
RANDOM_REPS = 20
N_PAIRS = None          # None = all residential origins (population-weighted)
SEED = 42
FLUSH_EVERY = 25


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        line = "[{}] {}".format(stamp, msg)
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode(), flush=True)
        self.f.write("[{}] {}\n".format(stamp, msg))
        self.f.flush()


# ----------------------------------------------------------------------------
def stage_export(args, log):
    exp = os.path.join(HERE, "graph_exports", args.city)
    man = os.path.join(exp, "manifest.json")
    if os.path.exists(man) and not args.reexport:
        log("Stage 1 export: already present, skipped ({})".format(exp))
    else:
        if not args.password:
            raise SystemExit("Password needed for the export: -p <password>")
        cmd = [sys.executable, os.path.join(HERE, "export_neo4j_city.py"),
               "--city", args.city, "-n", args.uri, "-u", args.user,
               "-p", args.password]
        if args.database:
            cmd += ["--database", args.database]
        log("Stage 1 export: running", " ".join(cmd[:4]), "...")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit("Export failed - send me "
                             "graph_exports/{}/export_log.txt".format(args.city))
    with open(man, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("errors"):
        raise SystemExit("Export has errors: {} - send me export_log.txt"
                         .format(m["errors"]))
    return exp


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------
class Prepared:
    pass


def stage_prepare(args, exp, out, log):
    P = Prepared()
    log("Stage 2 prepare: loading graph (LTS variant main) ...")
    city = CityGraph(exp, "main", verbose=False)
    P.cities = {"main": city}
    log("  junctions {:,}  active ROUTE edges {:,}  physical segments {:,}  "
        "roads with PageRank {:,}  essential POIs {:,}".format(
            city.n, city.m, city.n_seg, len(city.road_pagerank),
            len(city.pois)))
    cand = city.residential_origins()
    pop_csv = os.path.join(ROOT, "data", "{}_population.csv".format(args.city))
    pop_rows, inside = odm.read_population(pop_csv, city)
    w_node = odm.node_population(city, cand, pop_rows[inside])
    P.w_node = w_node
    core = Core.from_city(city, w_node)
    layers = {m: Layer(core, m) for m in LAYERS}
    master_path = os.path.join(out, "od_master.csv")
    pois_path = os.path.join(out, "pois.pkl")
    if os.path.exists(master_path) and os.path.exists(pois_path):
        with open(master_path, encoding="utf-8") as f:
            master = list(csv.DictReader(f))
        for r in master:
            for k in ("pair", "origin_node", "origin_jid", "poi_index",
                      "poi_id"):
                r[k] = int(r[k])
            r["origin_weight"] = float(r["origin_weight"])
        with open(pois_path, "rb") as f:
            pois = pickle.load(f)
        log("  OD master loaded ({} pairs)".format(len(master)))
    else:
        master, pois = odm.build_master(city, layers, cand, w_node,
                                        n_pairs=N_PAIRS, seed=SEED, log=log)
        write_csv(master_path, master)
        with open(pois_path, "wb") as f:
            pickle.dump(pois, f)
        log("  OD master: {} pairs (A {}, B {})".format(
            len(master), sum(r["set"] == "A" for r in master),
            sum(r["set"] == "B" for r in master)))
    P.master, P.pois = master, pois

    alts, missing = cl.alternatives(city, 50)
    P.alts = alts
    P.named = cl.all_named_roads(city)
    rows = []
    for rank, (r, pr) in enumerate(alts, 1):
        E = city.road_edges[r]
        rows.append({"rank": rank, "road_name": r, "pagerank": pr,
                     "edges": len(E),
                     "segments": len(np.unique(city.seg[E])),
                     "highway_mode": _mode([city.highway[e] for e in E])})
    write_csv(os.path.join(out, "alternatives.csv"), rows)
    if missing:
        log("  WARNING {} top-half roads have no ROUTE edge with that name "
            "and are skipped: {}".format(len(missing), missing[:10]))
    info = {
        "city": args.city, "bbox": city.bbox,
        "junctions": city.n, "route_edges": city.m,
        "physical_segments": city.n_seg,
        "candidate_origins": int(len(cand)),
        "census_sections_total": int(len(pop_rows)),
        "census_sections_in_area": int(inside.sum()),
        "residents_municipality": float(pop_rows[:, 2].sum()),
        "residents_in_area": float(pop_rows[inside, 2].sum()),
        "alternatives": len(alts), "alternatives_missing": missing,
        "named_roads_pool": len(P.named),
        "pois_total": len(city.pois),
        "pois_polygons": sum(1 for q in city.pois if q["geom"] == "way"),
        "pois_usable": len(pois),
    }
    with open(os.path.join(out, "prepare.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    log("  alternatives (top half by PageRank): {}; residents in study area "
        "{:,.0f} of {:,.0f}".format(len(alts), info["residents_in_area"],
                                    info["residents_municipality"]))
    return P


def _mode(vals):
    if not vals:
        return ""
    u, c = np.unique(np.array(vals, dtype=object).astype(str),
                     return_counts=True)
    return str(u[int(np.argmax(c))])


# ----------------------------------------------------------------------------
def stage_check(args, exp, out, P, log):
    """Old OD pairs + old settings must reproduce the old Neo4j baseline."""
    base = OLD_RUNS.get(args.city)
    jpath = os.path.join(ROOT, base + ".json") if base else None
    opath = os.path.join(ROOT, base + "_od_pairs.json") if base else None
    res_path = os.path.join(out, "check_legacy.json")
    if os.path.exists(res_path):
        log("Stage 3 check: already done, skipped")
        return
    if not (jpath and os.path.exists(jpath) and os.path.exists(opath)):
        log("Stage 3 check: old result files not found - skipped")
        return
    log("Stage 3 check: reproducing the old baseline ...")
    with open(jpath, encoding="utf-8", errors="replace") as f:
        old = json.load(f)
    with open(opath, encoding="utf-8", errors="replace") as f:
        old_od = json.load(f)
    city = CityGraph(exp, "stored", verbose=False)
    try:
        o = np.array([city.jindex[x["origin_id"]] for x in old_od])
        d = np.array([city.jindex[x["dest_junction_id"]] for x in old_od])
    except KeyError:
        msg = ("node ids of the old OD pairs are not in the database "
               "(database rebuilt since July?) - check skipped")
        log("  WARNING", msg)
        with open(res_path, "w") as f:
            json.dump({"skipped": msg}, f)
        return
    core = Core.from_city(city, P.w_node)
    sim = Simulator(core, {m: (o, d) for m in LAYERS}, "length")
    got = {m: int(np.isfinite(sim.dist0[m]).sum()) for m in LAYERS}
    exp_ = {"car": old["baseline"]["car_reachable"],
            "bike_safe": old["baseline"]["bike_safe_reachable"],
            "bike_all": old["baseline"]["bike_all_reachable"]}
    log("  baseline reachable pairs  offline {}  |  Neo4j {}".format(got, exp_))
    # a few single closures with the legacy closure rule
    old_res = {r["road_name"]: r for r in old["results"]
               if "road_name" in r and "car" in r}
    rows = []
    for road, _ in P.alts[:15]:
        if road not in old_res or road not in city.road_edges:
            continue
        mask, ctype, ne, ns = cl.single_closure(city, road, 0.25,
                                                legacy=True)
        chg, _ = sim.run(mask, frag=False)
        row = {"road": road, "edges_closed_offline": ne,
               "edges_closed_neo4j": old_res[road].get("edges_disrupted")}
        for m in LAYERS:
            new = sim.dist0[m].copy()
            for (mm, i, dv, _, _) in chg:
                if mm == m:
                    new[i] = dv
            b = sim.dist0[m]
            lost = int((np.isfinite(b) & ~np.isfinite(new)).sum())
            row["{}_lost_offline".format(m)] = lost
            row["{}_lost_neo4j".format(m)] = old_res[road][m]["pois_lost"]
        rows.append(row)
    write_csv(os.path.join(out, "check_legacy_roads.csv"), rows)
    agree = sum(all(r["{}_lost_offline".format(m)] ==
                    r["{}_lost_neo4j".format(m)] for m in LAYERS)
                for r in rows)
    log("  legacy single closures: {} of {} roads identical in lost pairs "
        "(small differences can come from ties in the old edge "
        "selection)".format(agree, len(rows)))
    bad = any(abs(got[m] - exp_[m]) > 2 for m in LAYERS)
    with open(res_path, "w") as f:
        json.dump({"offline": got, "neo4j": exp_, "roads_identical": agree,
                   "roads_checked": len(rows)}, f, indent=2)
    if bad:
        log("  WARNING: offline baseline differs from the old Neo4j baseline "
            "by more than 2 pairs. The run continues; I will look at "
            "check_legacy.json.")
        return
    log("  check passed")


# ----------------------------------------------------------------------------
def _flush(vdir, buf_rows, buf_frag, done_ids):
    k = len([f for f in os.listdir(vdir) if f.startswith("changes_part")])
    with gzip.open(os.path.join(vdir, "changes_part{:04d}.csv.gz".format(k)),
                   "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "layer", "pair", "new_dist", "iso", "inc"])
        for r in buf_rows:
            w.writerow([r[0], r[1], r[2],
                        "" if not np.isfinite(r[3]) else round(r[3], 3),
                        int(r[4]), int(r[5])])
    if buf_frag:
        fields = ["scenario"] + [x for x in buf_frag[0] if x != "scenario"]
        write_csv(os.path.join(vdir, "frag_part{:04d}.csv".format(k)),
                  buf_frag, fields)
    with open(os.path.join(vdir, "done.txt"), "a") as f:
        for s in done_ids:
            f.write("{}\n".format(s))


def run_scenarios(vdir, scen, closed_of, workers, log, label):
    """scen: list of dict rows (with 'scenario'); closed_of(row) -> idx."""
    done = set()
    dpath = os.path.join(vdir, "done.txt")
    if os.path.exists(dpath):
        with open(dpath) as f:
            done = {int(x) for x in f.read().split()}
    todo = [s for s in scen if int(s["scenario"]) not in done]
    if not todo:
        log("  {}: all {} scenarios already done".format(label, len(scen)))
        return
    log("  {}: {} scenarios to run ({} done before), {} worker(s)".format(
        label, len(todo), len(done), workers))
    tasks = [(int(s["scenario"]), closed_of(s)) for s in todo]
    init = (os.path.join(vdir, "core.npz"), os.path.join(vdir, "access.npz"),
            os.path.join(vdir, "baseline.pkl"), _read_meta(vdir)["cost"])
    t0 = time.time()
    buf_r, buf_f, buf_ids = [], [], []
    ndone = 0
    if workers > 1:
        pool = mp.get_context("spawn").Pool(workers, par.init_worker, init)
        it = pool.imap_unordered(par.run_task, tasks, chunksize=1)
    else:
        par.init_worker(*init)
        pool = None
        it = map(par.run_task, tasks)
    try:
        for sid, rows, frows in it:
            buf_r.extend(rows)
            buf_f.extend(frows)
            buf_ids.append(sid)
            ndone += 1
            if len(buf_ids) >= FLUSH_EVERY or ndone == len(tasks):
                _flush(vdir, buf_r, buf_f, buf_ids)
                buf_r, buf_f, buf_ids = [], [], []
                el = time.time() - t0
                eta = el / ndone * (len(tasks) - ndone)
                log("    {}: {}/{} done, elapsed {:.1f} min, remaining "
                    "~{:.1f} min".format(label, ndone, len(tasks), el / 60,
                                         eta / 60))
    finally:
        if buf_ids:
            _flush(vdir, buf_r, buf_f, buf_ids)
        if pool is not None:
            pool.close()
            pool.join()


def _read_meta(vdir):
    with open(os.path.join(vdir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def setup_variant(vdir, city, P, var, log):
    """Access nodes + baseline for one configuration (saved for workers)."""
    name, ltsv, cost, access, dest, depth, pairs = var
    os.makedirs(vdir, exist_ok=True)
    if os.path.exists(os.path.join(vdir, "baseline.pkl")):
        return
    sub = [r for r in P.master if pairs == "AB" or r["set"] in pairs]
    core = Core.from_city(city, P.w_node)
    acc = odm.access_nodes(city, {m: Layer(core, m) for m in LAYERS}, sub,
                           P.pois, mode=access, dest_rule=dest)
    sim = Simulator(core, acc, cost)
    core.save(os.path.join(vdir, "core.npz"))
    pair_ids = np.array([r["pair"] for r in sub], dtype=np.int64)
    np.savez_compressed(os.path.join(vdir, "access.npz"), pair_ids=pair_ids,
                        **{"o_" + m: acc[m][0] for m in LAYERS},
                        **{"d_" + m: acc[m][1] for m in LAYERS})
    with open(os.path.join(vdir, "baseline.pkl"), "wb") as f:
        pickle.dump(sim.baseline_state(), f, protocol=4)
    rows = []
    for k, r in enumerate(sub):
        row = {"pair": r["pair"], "set": r["set"],
               "weight": float(r["origin_weight"])}
        for m in LAYERS:
            row["o_" + m] = int(acc[m][0][k])
            row["d_" + m] = int(acc[m][1][k])
            dv = sim.dist0[m][k]
            row["dist0_" + m] = "" if not np.isfinite(dv) else round(dv, 3)
        rows.append(row)
    write_csv(os.path.join(vdir, "pairs.csv"), rows)
    meta = {"variant": name, "lts": ltsv, "cost": cost, "access": access,
            "dest": dest, "depth": depth, "pairs": pairs,
            "n_pairs": len(sub),
            "baseline_reachable": {m: int(np.isfinite(sim.dist0[m]).sum())
                                   for m in LAYERS},
            "created": _dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(vdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log("  [{}] pairs {}, baseline reachable {}".format(
        name, len(sub), meta["baseline_reachable"]))


def stage_single(args, exp, out, P, log, variants):
    for var in variants:
        name, ltsv, cost, access, dest, depth, pairs = var
        if ltsv not in P.cities:
            log("  loading graph with LTS variant '{}' ...".format(ltsv))
            P.cities[ltsv] = CityGraph(exp, ltsv, verbose=False)
            if ltsv != "main":
                a = np.nan_to_num(P.cities[ltsv].lts, nan=9) <= 2
                b = np.nan_to_num(P.cities["main"].lts, nan=9) <= 2
                changed = int((a != b).sum())
            else:
                changed = 0
            log("    edges whose low-stress status differs from main: "
                "{:,}".format(changed))
        city = P.cities[ltsv]
        vdir = os.path.join(out, "single_" + name)
        setup_variant(vdir, city, P, var, log)
        scen_path = os.path.join(vdir, "scenarios.csv")
        if not os.path.exists(scen_path):
            rows = []
            for sid, (road, pr) in enumerate(P.alts):
                mask, ctype, ne, ns = cl.single_closure(city, road, depth)
                rows.append({"scenario": sid, "kind": "single",
                             "road_name": road, "pagerank": pr,
                             "depth": depth, "closure_type": ctype,
                             "closed_edges": ne, "closed_segments": ns,
                             "road_segments": len(np.unique(
                                 city.seg[city.road_edges[road]]))})
            write_csv(scen_path, rows)
        with open(scen_path, encoding="utf-8") as f:
            scen = list(csv.DictReader(f))
        if args.smoke:
            scen = scen[:args.smoke]

        def closed_of(s, city=city, depth=depth):
            return np.where(cl.single_closure(city, s["road_name"],
                                              depth)[0])[0]
        run_scenarios(vdir, scen, closed_of, args.workers, log,
                      "single_" + name)


def stage_asym(args, exp, out, P, log):
    city = P.cities["main"]
    vdir = os.path.join(out, "asym")
    setup_variant(vdir, city, P, VARIANTS[0], log)
    scen_path = os.path.join(vdir, "scenarios.csv")
    if not os.path.exists(scen_path):
        alt_names = [r for r, _ in P.alts]
        rows, sid = [], 0
        for k in K_ASYM:
            rows.append({"scenario": sid, "kind": "targeted_pagerank",
                         "k": k, "rep": 0, "pool": "alternatives",
                         "roads": json.dumps(alt_names[:k],
                                             ensure_ascii=False)})
            sid += 1
            for pool_name, pool in (("all_named", P.named),
                                    ("alternatives", alt_names)):
                rng = np.random.default_rng(
                    [SEED, k, 1 if pool_name == "all_named" else 2])
                for rep in range(RANDOM_REPS):
                    pick = rng.choice(len(pool), size=min(k, len(pool)),
                                      replace=False)
                    rows.append({"scenario": sid, "kind": "random", "k": k,
                                 "rep": rep, "pool": pool_name,
                                 "roads": json.dumps(
                                     [pool[i] for i in sorted(pick)],
                                     ensure_ascii=False)})
                    sid += 1
        for r in rows:
            mask, ne, ns = cl.roads_closure(city, json.loads(r["roads"]))
            r["closed_edges"], r["closed_segments"] = ne, ns
        write_csv(scen_path, rows)
    with open(scen_path, encoding="utf-8") as f:
        scen = list(csv.DictReader(f))
    if args.smoke:
        scen = scen[:args.smoke]

    def closed_of(s):
        return np.where(cl.roads_closure(city, json.loads(s["roads"]))[0])[0]
    run_scenarios(vdir, scen, closed_of, args.workers, log, "asym")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--city", required=True,
                    choices=["modena", "ferrara", "pesaro"])
    ap.add_argument("-n", "--uri", default="bolt://localhost:7687")
    ap.add_argument("-u", "--user", default="neo4j")
    ap.add_argument("-p", "--password", default=None)
    ap.add_argument("--database", default=None)
    ap.add_argument("--workers", type=int,
                    default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--reexport", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="test run: only the first N scenarios per stage, "
                         "written to results_smoke/")
    ap.add_argument("--stages", default="export,prepare,single,asym")
    args = ap.parse_args()

    out = os.path.join(HERE, "results_smoke" if args.smoke else RESULTS_DIR,
                       args.city)
    os.makedirs(out, exist_ok=True)
    log = Log(os.path.join(out, "log.txt"))
    log("=" * 70)
    log("run_city.py {}  python {}  numpy {}  platform {}  workers {}".format(
        args.city, platform.python_version(), np.__version__,
        platform.platform(), args.workers))
    stages = [s.strip() for s in args.stages.split(",")]
    t0 = time.time()
    exp = stage_export(args, log) if "export" in stages else \
        os.path.join(HERE, "graph_exports", args.city)
    P = stage_prepare(args, exp, out, log)
    if "check" in stages:
        stage_check(args, exp, out, P, log)
    if "single" in stages:
        log("Stage 4 single-road closures")
        stage_single(args, exp, out, P, log, VARIANTS)
    if "asym" in stages:
        log("Stage 5 simultaneous closures (asymmetry)")
        stage_asym(args, exp, out, P, log)
    with open(os.path.join(out, "FINISHED.txt"), "w") as f:
        f.write("finished {} in {:.1f} h\n".format(
            _dt.datetime.now().isoformat(timespec="seconds"),
            (time.time() - t0) / 3600))
    log("ALL STAGES FINISHED in {:.1f} h. Results: {}".format(
        (time.time() - t0) / 3600, out))


if __name__ == "__main__":
    mp.freeze_support()
    main()
