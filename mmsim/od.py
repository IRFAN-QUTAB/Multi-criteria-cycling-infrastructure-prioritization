import csv

import numpy as np
from scipy.spatial import cKDTree

ACCESS_RADIUS = 100.0
N_CANDIDATES = 3


def read_population(path, city):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        la = cols.get("lat") or cols.get("latitude")
        lo = cols.get("lon") or cols.get("longitude")
        po = cols.get("population") or cols.get("pop") or cols.get("p1")
        for r in rd:
            try:
                rows.append((float(r[la]), float(r[lo]), float(r[po])))
            except (TypeError, ValueError):
                continue
    a = np.array(rows, dtype=float)
    s, w, nn, e = city.bbox
    inside = ((a[:, 0] >= s) & (a[:, 0] <= nn) & (a[:, 1] >= w)
              & (a[:, 1] <= e))
    return a, inside


def node_population(city, cand, pop_rows):
    """Population weight of every junction (non-zero only for candidates)."""
    sec_xy = city.proj(pop_rows[:, 0], pop_rows[:, 1])
    cand_xy = city.xy[cand]
    t_sec = cKDTree(sec_xy)
    _, near_sec = t_sec.query(cand_xy, k=1)
    count = np.bincount(near_sec, minlength=len(pop_rows))
    w = np.zeros(city.n)
    share = pop_rows[:, 2] / np.maximum(count, 1)
    w[cand] = share[near_sec]
    orphan = np.where(count == 0)[0]
    if len(orphan):
        t_c = cKDTree(cand_xy)
        _, nc = t_c.query(sec_xy[orphan], k=1)
        np.add.at(w, cand[nc], pop_rows[orphan, 2])
    return w


def efraimidis_spirakis(weights, n, seed):
    rng = np.random.default_rng(seed)
    w = np.maximum(np.asarray(weights, float), 1e-12)
    keys = rng.random(len(w)) ** (1.0 / w)
    order = np.argsort(-keys, kind="mergesort")
    return order[:min(n, len(w))]


class AccessFinder:
    """Nearest layer junction within ACCESS_RADIUS, main component first."""

    def __init__(self, city, layer):
        self.city = city
        self.nodes = np.where(layer.nodes & city.has_coords)[0]
        self.lcc = np.where(layer.lcc0 & city.has_coords)[0]
        self.t_all = cKDTree(city.xy[self.nodes]) if len(self.nodes) else None
        self.t_lcc = cKDTree(city.xy[self.lcc]) if len(self.lcc) else None
        self.in_lcc = layer.lcc0

    def _nearest(self, tree, ids, pts):
        if tree is None:
            return -1, np.inf
        d, i = tree.query(pts, k=1, distance_upper_bound=ACCESS_RADIUS)
        d = np.atleast_1d(d)
        i = np.atleast_1d(i)
        k = int(np.argmin(d))
        if not np.isfinite(d[k]):
            return -1, np.inf
        return int(ids[i[k]]), float(d[k])

    def for_points(self, pts_xy, own_node=None):
        if own_node is not None and own_node >= 0 and self.in_lcc[own_node]:
            return own_node
        j, _ = self._nearest(self.t_lcc, self.lcc, pts_xy)
        if j >= 0:
            return j
        j, _ = self._nearest(self.t_all, self.nodes, pts_xy)
        if j >= 0:
            return j
        return -1


def build_master(city, sim_layers, cand, w_node, n_pairs=None, seed=42,
                 log=print):
    """
    Sample origins, choose each origin's destination among its three
    Euclidean-nearest essential services by car network distance, and
    return the master OD table (independent of LTS / access variants).
    sim_layers: dict of network.Layer for the main configuration.
    cand: candidate origin junctions; w_node: population weight per junction.
    """
    from .network import route_pairs

    car_af = AccessFinder(city, sim_layers["car"])
    pois = []
    for q in city.pois:
        pxy = city.proj(q["points"][:, 0], q["points"][:, 1])
        jc = car_af.for_points(pxy)
        if jc < 0:
            continue
        q = dict(q)
        q["xy"] = pxy
        q["centroid"] = pxy.mean(axis=0)
        q["car_node"] = jc
        pois.append(q)
    log("  essential POIs with a car access junction within 100 m: "
        "{:,} of {:,}".format(len(pois), len(city.pois)))
    cent = np.array([q["centroid"] for q in pois])
    t_poi = cKDTree(cent)

    # three Euclidean-nearest services per candidate origin
    kq = min(12, len(pois))
    _, near = t_poi.query(city.xy[cand], k=kq)
    cands = []
    for ci, o in enumerate(cand):
        oxy = city.xy[o]
        lst = []
        for pi in np.atleast_1d(near[ci]):
            q = pois[int(pi)]
            if q["car_node"] == o:
                continue
            dmin = float(np.min(np.hypot(q["xy"][:, 0] - oxy[0],
                                         q["xy"][:, 1] - oxy[1])))
            lst.append((dmin, int(pi)))
        lst.sort()
        cands.append([pi for _, pi in lst[:N_CANDIDATES]])
    has = np.array([len(c) > 0 for c in cands])
    pool = np.where(has & (w_node[cand] > 0))[0]
    if n_pairs is None:
        # every residential origin, weighted by its population share
        pick = pool
    else:
        pick = pool[efraimidis_spirakis(w_node[cand][pool], n_pairs, seed)]

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(pick))
    sets = np.where(perm < len(pick) // 2, "A", "B")

    # destination = candidate with the smallest car network distance
    _, GT = sim_layers["car"].graphs(None)
    oo, dd, owner = [], [], []
    for r, ci in enumerate(pick):
        for pi in cands[ci]:
            oo.append(cand[ci])
            dd.append(pois[pi]["car_node"])
            owner.append((r, pi))
    dist, _ = route_pairs(GT, oo, dd)
    best = {}
    for (r, pi), dv in zip(owner, dist):
        if r not in best or dv < best[r][0]:
            best[r] = (dv, pi)
    rows = []
    for r, ci in enumerate(pick):
        dv, pi = best[r]
        if not np.isfinite(dv):          # none reachable: Euclidean nearest
            pi = cands[ci][0]
        q = pois[pi]
        rows.append({
            "pair": r, "set": sets[r],
            "origin_node": int(cand[ci]),
            "origin_jid": int(city.jid[cand[ci]]),
            "origin_lat": float(city.lat[cand[ci]]),
            "origin_lon": float(city.lon[cand[ci]]),
            "origin_weight": float(w_node[cand[ci]]),
            "poi_index": int(pi), "poi_id": int(q["poi_id"]),
            "amenity": q["amenity"], "poi_geom": q["geom"],
            "cand_poi_index": " ".join(str(x) for x in cands[ci]),
        })
    return rows, pois


def access_nodes(city, sim_layers, master, pois, mode="nearest100",
                 dest_rule="car"):
    """
    Access junction of every origin and destination in each layer.
    mode: "nearest100" (main) or "legacy" (same junction in all layers).
    dest_rule: "car" (service chosen by car distance, main) or "layer"
               (each layer goes to its own nearest of the three candidates).
    """
    from .network import route_pairs

    finders = {m: AccessFinder(city, sim_layers[m]) for m in sim_layers}
    out = {}
    o_main = np.array([r["origin_node"] for r in master], dtype=np.int64)
    for m in sim_layers:
        if mode == "legacy":
            o = o_main.copy()
            d = np.array([pois[r["poi_index"]]["car_node"] for r in master])
            out[m] = (o, d)
            continue
        f = finders[m]
        o = []
        for x in o_main:
            j = f.for_points(city.xy[[x]], own_node=int(x))
            o.append(j if j >= 0 else int(x))
        o = np.array(o, dtype=np.int64)

        def poi_node(pi):
            q = pois[pi]
            j = f.for_points(q["xy"])
            return j if j >= 0 else q["car_node"]

        if dest_rule == "car" or m == "car":
            d = np.array([poi_node(r["poi_index"]) for r in master])
        else:
            _, GT = sim_layers[m].graphs(None)
            oo, dd, owner = [], [], []
            for r, row in enumerate(master):
                for pi in [int(x) for x in row["cand_poi_index"].split()]:
                    oo.append(o[r])
                    dd.append(poi_node(pi))
                    owner.append(r)
            dist, _ = route_pairs(GT, oo, dd)
            best = {}
            for r, nd, dv in zip(owner, dd, dist):
                if r not in best or dv < best[r][0]:
                    best[r] = (dv, nd)
            d = np.array([best[r][1] if np.isfinite(best[r][0])
                          else poi_node(master[r]["poi_index"])
                          for r in range(len(master))])
        out[m] = (o, np.asarray(d, dtype=np.int64))
    return out
