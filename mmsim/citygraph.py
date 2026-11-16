
import ast
import gzip
import json
import math
import os

import numpy as np

from . import lts as ltsmod

AMENITY_TYPES = {'hospital', 'clinic', 'doctors', 'pharmacy', 'dentist',
                 'school', 'kindergarten', 'university', 'college',
                 'police', 'post_office', 'fire_station', 'library', 'bank'}
SHOP_TYPES = {'supermarket', 'convenience'}
RESIDENTIAL_TYPES = {'residential', 'living_street'}
MIN_EDGE_LENGTH = 1e-3   # metres; zero-length edges would vanish in CSR


def _iter_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _names(value):
    """ROUTE.name as a set of strings (lists are kept as several names)."""
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(x) for x in value if x is not None}
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            lst = ast.literal_eval(value.strip())
            if isinstance(lst, list):
                return {str(x) for x in lst if x is not None}
        except (ValueError, SyntaxError):
            pass
    return {str(value)}


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


class LocalXY:
    """Equirectangular projection in metres around a reference latitude."""

    def __init__(self, lat0):
        self.kx = 111320.0 * math.cos(math.radians(lat0))
        self.ky = 110540.0

    def __call__(self, lat, lon):
        return np.column_stack([np.asarray(lon, float) * self.kx,
                                np.asarray(lat, float) * self.ky])


class CityGraph:
    def __init__(self, export_dir, lts_variant="main", verbose=True):
        self.export_dir = export_dir
        self.lts_variant = lts_variant
        log = print if verbose else (lambda *a, **k: None)

        node_props, node_labels = {}, {}
        for rec in _iter_jsonl(os.path.join(export_dir, "nodes.jsonl.gz")):
            node_props[rec["id"]] = rec["p"] or {}
            node_labels[rec["id"]] = set(rec["labels"] or [])

        # ---------------- junctions ----------------
        jids = sorted(i for i, l in node_labels.items() if "RoadJunction" in l)
        self.jid = np.array(jids, dtype=np.int64)
        self.jindex = {j: k for k, j in enumerate(jids)}
        n = len(jids)
        self.n = n
        lat = np.full(n, np.nan)
        lon = np.full(n, np.nan)
        bc = np.zeros(n)
        for k, j in enumerate(jids):
            p = node_props[j]
            la, lo = p.get("lat"), p.get("lon")
            if la is None and p.get("y") is not None:
                la = p.get("y")
            if lo is None and p.get("x") is not None:
                lo = p.get("x")
            lat[k] = _num(la)
            lon[k] = _num(lo)
            v = _num(p.get("car_bc"))
            bc[k] = v if np.isfinite(v) else 0.0
        self.lat, self.lon, self.car_bc = lat, lon, bc
        ok = ~np.isnan(lat)
        self.lat0 = float(np.nanmean(lat))
        self.proj = LocalXY(self.lat0)
        self.xy = self.proj(np.where(ok, lat, self.lat0),
                            np.where(ok, lon, 0.0))
        self.has_coords = ok
        self.bbox = (float(np.nanmin(lat)), float(np.nanmin(lon)),
                     float(np.nanmax(lat)), float(np.nanmax(lon)))

        # ---------------- relationships ----------------
        e_u, e_v, e_len, e_id, e_props = [], [], [], [], []
        tags_of, part_of = {}, {}
        roads = []
        connected = []
        for rec in _iter_jsonl(os.path.join(export_dir, "rels.jsonl.gz")):
            t = rec["type"]
            s, d, p = rec["s"], rec["t"], rec["p"] or {}
            if t == "ROUTE":
                if s not in self.jindex or d not in self.jindex:
                    continue
                if str(p.get("status", "active")) != "active":
                    continue
                L = p.get("distance", p.get("length"))
                try:
                    L = float(L)
                except (TypeError, ValueError):
                    L = 1.0     # same fallback as the Neo4j version
                if not np.isfinite(L):
                    L = 1.0
                e_u.append(self.jindex[s])
                e_v.append(self.jindex[d])
                e_len.append(max(L, MIN_EDGE_LENGTH))
                e_id.append(rec["id"])
                e_props.append(p)
            elif t == "TAGS":
                tags_of.setdefault(s, []).append(d)
            elif t == "PART_OF":
                part_of.setdefault(d, []).append(s)
            elif t == "CONNECTED":
                connected.append((s, d))

        m = len(e_u)
        self.m = m
        self.u = np.array(e_u, dtype=np.int64)
        self.v = np.array(e_v, dtype=np.int64)
        self.length = np.array(e_len, dtype=float)
        self.edge_id = np.array(e_id, dtype=np.int64)
        self.props = e_props
        self.highway = np.array([str(ltsmod.first(p.get("highway")))
                                 for p in e_props], dtype=object)
        self.driveable = np.array([ltsmod.as_bool(p.get("driveable"))
                                   for p in e_props])
        self.bike_allowed = np.array([ltsmod.as_bool(p.get("bike_allowed"))
                                      for p in e_props])
        self.lts_stored = np.array(
            [np.nan if not ltsmod.as_bool(p.get("bike_allowed"))
             else _num(p.get("lts")) for p in e_props])
        self.lts = np.array(
            [np.nan if (x := ltsmod.edge_lts(p, lts_variant)) is None
             else float(x) for p in e_props])
        self.speed_tagged = np.array([ltsmod.speed_is_tagged(p.get("maxspeed"))
                                      for p in e_props])
        self.on_bus = np.array([ltsmod.as_bool(p.get("on_bus_route"))
                                for p in e_props])
        self.names = [_names(p.get("name")) for p in e_props]
        a = np.minimum(self.u, self.v)
        b = np.maximum(self.u, self.v)
        _, self.seg = np.unique(a * n + b, return_inverse=True)
        self.n_seg = int(self.seg.max()) + 1 if m else 0

        # ---------------- roads (dual graph nodes) ----------------
        self.road_pagerank = {}
        self.road_node = {}
        for i, labs in node_labels.items():
            if "RoadOsm" in labs:
                p = node_props[i]
                nm = p.get("name")
                if nm is None:
                    continue
                pr = _num(p.get("dual_pagerank"))
                self.road_node[str(nm)] = i
                if np.isfinite(pr):
                    self.road_pagerank[str(nm)] = pr
        self.road_edges = {}
        for k, ns in enumerate(self.names):
            for nm in ns:
                self.road_edges.setdefault(nm, []).append(k)
        self.road_edges = {k: np.array(v, dtype=np.int64)
                           for k, v in self.road_edges.items()}
        self.connected = connected

        # ---------------- essential POIs ----------------
        pois = []
        for i, labs in node_labels.items():
            if "POI" not in labs:
                continue
            kind = None
            for t_id in tags_of.get(i, []):
                tp = node_props.get(t_id, {})
                am, sh = tp.get("amenity"), tp.get("shop")
                if am in AMENITY_TYPES:
                    kind = am
                    break
                if sh in SHOP_TYPES:
                    kind = sh
                    break
            if kind is None:
                p = node_props[i]
                if p.get("amenity") in AMENITY_TYPES:
                    kind = p.get("amenity")
                elif p.get("shop") in SHOP_TYPES:
                    kind = p.get("shop")
                elif p.get("poi_type") in AMENITY_TYPES | SHOP_TYPES:
                    kind = p.get("poi_type")
            if kind is None:
                continue
            p = node_props[i]
            if np.isfinite(_num(p.get("lat"))) and np.isfinite(
                    _num(p.get("lon"))):
                pts = [(_num(p["lat"]), _num(p["lon"]))]
                geom = "node"
            else:
                pts = []
                for mem in part_of.get(i, []):
                    q = node_props.get(mem, {})
                    la, lo = _num(q.get("lat")), _num(q.get("lon"))
                    if np.isfinite(la) and np.isfinite(lo):
                        pts.append((la, lo))
                geom = "way"
            if not pts:
                continue
            pois.append({"poi_id": i, "amenity": kind, "geom": geom,
                         "points": np.array(pts, dtype=float)})
        self.pois = pois
        log("  [{}] junctions {:,}, active ROUTE edges {:,}, physical "
            "segments {:,}, roads with PageRank {:,}, essential POIs {:,} "
            "({} polygons)".format(
                os.path.basename(export_dir), n, m, self.n_seg,
                len(self.road_pagerank), len(pois),
                sum(1 for q in pois if q["geom"] == "way")))

    # ------------------------------------------------------------------
    def layer_mask(self, layer):
        if layer == "car":
            return self.driveable.copy()
        if layer == "bike_all":
            return self.bike_allowed.copy()
        if layer == "bike_safe":
            return self.bike_allowed & (self.lts <= 2)
        raise ValueError(layer)

    def residential_origins(self):
        """Junctions with an outgoing residential / living_street edge."""
        is_res = np.array([h in RESIDENTIAL_TYPES for h in self.highway])
        cand = np.unique(self.u[is_res])
        return cand[self.has_coords[cand]]

    def edges_of_roads(self, road_names):
        idx = [self.road_edges[r] for r in road_names if r in self.road_edges]
        if not idx:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(idx))
