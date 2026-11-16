"""
export_neo4j_city.py - read-only export of a city graph from Neo4j.

Stores every node and relationship with all properties, so that the
simulation, the MCDM analysis and the validation run offline.

    python export_neo4j_city.py --city modena -n bolt://localhost:7687 -u neo4j -p <password>

Optional: --database <name> if the city is not in the default database.

Output: graph_exports/<city>/
    nodes.jsonl.gz   one JSON object per node  {id, eid, labels, p}
    rels.jsonl.gz    one JSON object per rel   {id, type, s, t, p}
    manifest.json    counts, property keys, versions, checks, sha256
    export_log.txt   log of the run
"""

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import math
import os
import sys
import time

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    print("ERROR: the 'neo4j' Python driver is not installed in this "
          "environment. Use the same virtual environment as for "
          "vulnerability_analysis.py (pip install neo4j).")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
FETCH_SIZE = 5000

# What the offline pipeline needs. Missing items are reported as errors.
REQUIRED_LABELS = ["RoadJunction", "RoadOsm", "POI"]
REQUIRED_RELS = ["ROUTE", "NEAR_CAR", "CONNECTED"]
REQUIRED_ROUTE_KEYS = ["driveable", "bike_allowed", "lts", "distance",
                       "name", "highway", "status"]
REQUIRED_JUNCTION_KEYS = ["lat", "lon", "car_bc"]
REQUIRED_ROADOSM_KEYS = ["name", "dual_pagerank"]
# Raw OSM tags used to recompute LTS variants (warn only if absent).
USEFUL_ROUTE_KEYS = ["maxspeed", "lanes", "speed", "cycleway",
                     "cycleway:left", "cycleway:right", "cycleway:both",
                     "bicycle", "segregated", "on_bus_route", "oneway",
                     "bicycle_class", "route_type", "osmid", "length",
                     "access", "motor_vehicle", "surface", "service",
                     "tracktype", "junction"]


class Tee:
    """Write console output to the log file as well."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode(), flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def to_jsonable(v):
    """Convert Neo4j values (points, temporal types, bytes) to JSON."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    # neo4j.spatial points expose x, y (and srid)
    if hasattr(v, "x") and hasattr(v, "y"):
        out = {"_point": True, "x": float(v.x), "y": float(v.y)}
        if hasattr(v, "srid"):
            out["srid"] = v.srid
        return out
    # neo4j.time types and anything else
    return str(v)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_list(session, q, **params):
    return [r.data() for r in session.run(q, **params)]


def main():
    ap = argparse.ArgumentParser(description="Full Neo4j export of one city")
    ap.add_argument("--city", required=True,
                    help="modena | ferrara | pesaro (used as folder name)")
    ap.add_argument("-n", "--uri", default="bolt://localhost:7687")
    ap.add_argument("-u", "--user", default="neo4j")
    ap.add_argument("-p", "--password", required=True)
    ap.add_argument("--database", default=None,
                    help="database name (default: server default)")
    ap.add_argument("--out", default=os.path.join(HERE, "graph_exports"))
    args = ap.parse_args()

    city = args.city.strip().lower()
    out_dir = os.path.join(args.out, city)
    os.makedirs(out_dir, exist_ok=True)
    log = Tee(os.path.join(out_dir, "export_log.txt"))
    t0 = time.time()
    log("Export of city '{}' started {}".format(
        city, _dt.datetime.now().isoformat(timespec="seconds")))
    log("Output folder:", out_dir)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    driver.verify_connectivity()
    sess_kw = {"fetch_size": FETCH_SIZE}
    if args.database:
        sess_kw["database"] = args.database

    manifest = {"city": city, "uri": args.uri, "database": args.database,
                "exported_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "errors": [], "warnings": []}

    with driver.session(**sess_kw) as s:
        # ---------------- versions ----------------
        try:
            manifest["neo4j_components"] = run_list(
                s, "CALL dbms.components() YIELD name, versions, edition "
                   "RETURN name, versions, edition")
        except Exception as e:  # noqa: BLE001
            manifest["warnings"].append("dbms.components failed: {}".format(e))
        try:
            manifest["gds_version"] = run_list(
                s, "RETURN gds.version() AS v")[0]["v"]
        except Exception:  # noqa: BLE001
            manifest["gds_version"] = None

        # ---------------- counts ----------------
        labels = run_list(s, "MATCH (n) UNWIND labels(n) AS l "
                             "RETURN l AS label, count(*) AS n ORDER BY l")
        rels = run_list(s, "MATCH ()-[r]->() RETURN type(r) AS type, "
                           "count(*) AS n ORDER BY type")
        n_nodes = run_list(s, "MATCH (n) RETURN count(n) AS n")[0]["n"]
        n_rels = run_list(s, "MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
        manifest["label_counts"] = {d["label"]: d["n"] for d in labels}
        manifest["rel_type_counts"] = {d["type"]: d["n"] for d in rels}
        manifest["n_nodes"], manifest["n_rels"] = n_nodes, n_rels
        log("Nodes: {:,}  Relationships: {:,}".format(n_nodes, n_rels))
        log("Labels:", manifest["label_counts"])
        log("Relationship types:", manifest["rel_type_counts"])

        # ---------------- property keys ----------------
        manifest["route_keys"] = {d["k"]: d["n"] for d in run_list(
            s, "MATCH ()-[r:ROUTE]->() UNWIND keys(r) AS k "
               "RETURN k, count(*) AS n ORDER BY k")}
        manifest["junction_keys"] = {d["k"]: d["n"] for d in run_list(
            s, "MATCH (n:RoadJunction) UNWIND keys(n) AS k "
               "RETURN k, count(*) AS n ORDER BY k")}
        manifest["roadosm_keys"] = {d["k"]: d["n"] for d in run_list(
            s, "MATCH (n:RoadOsm) UNWIND keys(n) AS k "
               "RETURN k, count(*) AS n ORDER BY k")}
        manifest["poi_keys"] = {d["k"]: d["n"] for d in run_list(
            s, "MATCH (n:POI) UNWIND keys(n) AS k "
               "RETURN k, count(*) AS n ORDER BY k")}
        log("ROUTE property keys:", sorted(manifest["route_keys"]))

        # ---------------- checks ----------------
        for lab in REQUIRED_LABELS:
            if manifest["label_counts"].get(lab, 0) == 0:
                manifest["errors"].append("label missing: {}".format(lab))
        for rt in REQUIRED_RELS:
            if manifest["rel_type_counts"].get(rt, 0) == 0:
                manifest["errors"].append("relationship missing: {}".format(rt))
        for k in REQUIRED_ROUTE_KEYS:
            if k not in manifest["route_keys"]:
                manifest["errors"].append("ROUTE property missing: {}".format(k))
        for k in REQUIRED_JUNCTION_KEYS:
            if k not in manifest["junction_keys"]:
                manifest["errors"].append(
                    "RoadJunction property missing: {}".format(k))
        for k in REQUIRED_ROADOSM_KEYS:
            if k not in manifest["roadosm_keys"]:
                manifest["errors"].append(
                    "RoadOsm property missing: {}".format(k))
        for k in USEFUL_ROUTE_KEYS:
            if k not in manifest["route_keys"]:
                manifest["warnings"].append(
                    "ROUTE property not present (ok if never tagged): "
                    "{}".format(k))
        n_dis = run_list(s, "MATCH ()-[r:ROUTE]->() WHERE r.disrupted = true "
                            "RETURN count(r) AS n")[0]["n"]
        manifest["route_flagged_disrupted"] = n_dis
        if n_dis:
            manifest["warnings"].append(
                "{} ROUTE edges still carry disrupted=true from an earlier "
                "run; they are exported as they are and the offline "
                "pipeline ignores the flag".format(n_dis))
        tag_types = run_list(
            s, "MATCH (p:POI)-[:TAGS]->(t:Tag) "
               "RETURN count(DISTINCT p) AS pois, "
               "count(DISTINCT t.amenity) AS amenity_values, "
               "count(DISTINCT t.shop) AS shop_values")
        manifest["poi_tag_summary"] = tag_types[0] if tag_types else None

        for e in manifest["errors"]:
            log("ERROR:", e)
        for w in manifest["warnings"]:
            log("warning:", w)

        # ---------------- nodes ----------------
        nodes_path = os.path.join(out_dir, "nodes.jsonl.gz")
        log("Exporting nodes ...")
        cnt = 0
        try:  # elementId() exists from Neo4j 5; fall back for 4.x
            run_list(s, "RETURN elementId(null) AS x")
            eid_expr = "elementId(n)"
        except Exception:  # noqa: BLE001
            eid_expr = "toString(id(n))"
        res = s.run("MATCH (n) RETURN id(n) AS id, {} AS eid, "
                    "labels(n) AS labels, properties(n) AS p".format(eid_expr))
        with gzip.open(nodes_path, "wt", encoding="utf-8") as f:
            for rec in res:
                f.write(json.dumps({"id": rec["id"], "eid": rec["eid"],
                                    "labels": rec["labels"],
                                    "p": to_jsonable(rec["p"])},
                                   ensure_ascii=False) + "\n")
                cnt += 1
                if cnt % 50000 == 0:
                    log("  nodes written: {:,}".format(cnt))
        manifest["nodes_written"] = cnt
        log("  nodes written: {:,} (expected {:,})".format(cnt, n_nodes))

        # ---------------- relationships ----------------
        rels_path = os.path.join(out_dir, "rels.jsonl.gz")
        log("Exporting relationships ...")
        cnt = 0
        res = s.run("MATCH (a)-[r]->(b) RETURN id(r) AS id, type(r) AS type, "
                    "id(a) AS s, id(b) AS t, properties(r) AS p")
        with gzip.open(rels_path, "wt", encoding="utf-8") as f:
            for rec in res:
                f.write(json.dumps({"id": rec["id"], "type": rec["type"],
                                    "s": rec["s"], "t": rec["t"],
                                    "p": to_jsonable(rec["p"])},
                                   ensure_ascii=False) + "\n")
                cnt += 1
                if cnt % 100000 == 0:
                    log("  relationships written: {:,}".format(cnt))
        manifest["rels_written"] = cnt
        log("  relationships written: {:,} (expected {:,})".format(cnt, n_rels))

    driver.close()

    if manifest["nodes_written"] != manifest["n_nodes"]:
        manifest["errors"].append("node count mismatch")
    if manifest["rels_written"] != manifest["n_rels"]:
        manifest["errors"].append("relationship count mismatch")
    manifest["sha256"] = {"nodes.jsonl.gz": sha256_of(nodes_path),
                          "rels.jsonl.gz": sha256_of(rels_path)}
    manifest["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log("Done in {:.0f} s.".format(manifest["seconds"]))
    if manifest["errors"]:
        log("FINISHED WITH ERRORS - send me export_log.txt:")
        for e in manifest["errors"]:
            log("  -", e)
    else:
        log("OK - export complete and consistent.")
    log.close()


if __name__ == "__main__":
    main()
