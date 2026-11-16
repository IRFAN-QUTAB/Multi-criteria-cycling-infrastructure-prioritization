"""
Routing core of the offline simulation (scipy.sparse.csgraph).

A Core holds only numeric arrays (fast to pickle / load in worker processes):
    n, u, v, length, driveable, bike_allowed, lts, pop
A Layer is one mode layer of the multiplex graph; closures are boolean masks
over the city edges, and a layer simply ignores its closed edges, exactly as
the Neo4j projections ignored flagged edges.

Shortest paths are computed from the destinations on the reversed graph, so
that one Dijkstra run serves all origins of a destination. After a closure,
only pairs whose baseline shortest path uses a closed edge are re-routed: a
closure can only remove edges, so every other pair keeps its baseline cost
exactly (this is exact, not an approximation).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

LAYERS = ("car", "bike_all", "bike_safe")
LTS_COST = {1: 1.0, 2: 1.5, 3: 3.0, 4: 5.0}    # Broach et al. (2012)
CHUNK = 64                                       # Dijkstra sources per call


class Core:
    def __init__(self, n, u, v, length, driveable, bike_allowed, lts, pop):
        self.n = int(n)
        self.u = np.asarray(u, dtype=np.int64)
        self.v = np.asarray(v, dtype=np.int64)
        self.length = np.asarray(length, dtype=float)
        self.driveable = np.asarray(driveable, dtype=bool)
        self.bike_allowed = np.asarray(bike_allowed, dtype=bool)
        self.lts = np.asarray(lts, dtype=float)
        self.pop = np.asarray(pop, dtype=float)
        self.m = len(self.u)

    @classmethod
    def from_city(cls, city, pop):
        return cls(city.n, city.u, city.v, city.length, city.driveable,
                   city.bike_allowed, city.lts, pop)

    def save(self, path):
        np.savez_compressed(path, n=self.n, u=self.u, v=self.v,
                            length=self.length, driveable=self.driveable,
                            bike_allowed=self.bike_allowed, lts=self.lts,
                            pop=self.pop)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(int(z["n"]), z["u"], z["v"], z["length"], z["driveable"],
                   z["bike_allowed"], z["lts"], z["pop"])

    def layer_mask(self, layer):
        if layer == "car":
            return self.driveable.copy()
        if layer == "bike_all":
            return self.bike_allowed.copy()
        if layer == "bike_safe":
            return self.bike_allowed & (self.lts <= 2)
        raise ValueError(layer)


class Layer:
    def __init__(self, core, name, cost_mode="length", extra_mask=None,
                 lts_override=None):
        """extra_mask / lts_override: used by the investment experiment
        (edges upgraded to low-stress)."""
        self.core, self.name, self.n = core, name, core.n
        lts = core.lts if lts_override is None else lts_override
        if name == "car":
            mask = core.driveable.copy()
        elif name == "bike_all":
            mask = core.bike_allowed.copy()
        else:
            mask = core.bike_allowed & (lts <= 2)
        if extra_mask is not None:
            mask = mask | extra_mask
        idx = np.where(mask)[0]
        cost = core.length[idx].copy()
        if cost_mode == "lts" and name != "car":
            mult = np.array([LTS_COST.get(int(x), 5.0) if np.isfinite(x)
                             else 5.0 for x in lts[idx]])
            cost = cost * mult
        order = np.lexsort((cost, core.v[idx], core.u[idx]))
        self.e = idx[order]
        self.c = cost[order]
        self.eu = core.u[self.e]
        self.ev = core.v[self.e]
        self.key = self.eu * self.n + self.ev
        self.mask = mask
        deg = (np.bincount(self.eu, minlength=self.n)
               + np.bincount(self.ev, minlength=self.n))
        self.nodes = deg > 0
        G, _ = self.graphs(None)
        lab = connected_components(G, directed=True, connection="weak")[1]
        cnt = np.bincount(lab[self.nodes]) if self.nodes.any() else [0]
        self.lcc0 = (lab == int(np.argmax(cnt))) & self.nodes

    def graphs(self, closed):
        """Forward and reversed CSR graphs without the closed edges.
        Parallel edges keep their minimum cost."""
        if closed is None:
            keep = np.ones(len(self.e), dtype=bool)
        else:
            keep = ~closed[self.e]
        key, c = self.key[keep], self.c[keep]
        if len(key):
            first = np.ones(len(key), dtype=bool)
            first[1:] = key[1:] != key[:-1]
            key, c = key[first], c[first]
        uu, vv = key // self.n, key % self.n
        G = csr_matrix((c, (uu, vv)), shape=(self.n, self.n))
        GT = csr_matrix((c, (vv, uu)), shape=(self.n, self.n))
        return G, GT

    def closed_keys(self, closed):
        if closed is None:
            return np.zeros(0, dtype=np.int64)
        return np.unique(self.key[closed[self.e]])

    def degree_after(self, closed):
        keep = ~closed[self.e]
        return (np.bincount(self.eu[keep], minlength=self.n)
                + np.bincount(self.ev[keep], minlength=self.n))


def dist_to_targets(GT, targets, origins_per_target, want_paths=False):
    """
    targets: unique target nodes; origins_per_target: list of origin arrays.
    Returns (dist list aligned with origins_per_target, path list or None).
    A path is the array of forward edge keys o->...->target.
    """
    n = GT.shape[0]
    dists, paths = [], [] if want_paths else None
    for c0 in range(0, len(targets), CHUNK):
        tg = np.asarray(targets[c0:c0 + CHUNK])
        if want_paths:
            D, P = dijkstra(GT, directed=True, indices=tg,
                            return_predecessors=True)
        else:
            D = dijkstra(GT, directed=True, indices=tg)
        for r, t in enumerate(tg):
            orgs = np.asarray(origins_per_target[c0 + r], dtype=np.int64)
            dists.append(D[r, orgs])
            if want_paths:
                pl = []
                for o in orgs:
                    if not np.isfinite(D[r, o]) or o == t:
                        pl.append(np.zeros(0, dtype=np.int64))
                        continue
                    keys = []
                    x = int(o)
                    guard = 0
                    while x != t:
                        nx = int(P[r, x])
                        if nx < 0:
                            keys = []
                            break
                        keys.append(x * n + nx)
                        x = nx
                        guard += 1
                        if guard > n:
                            raise RuntimeError("path reconstruction loop")
                    pl.append(np.array(keys, dtype=np.int64))
                paths.append(pl)
    return dists, paths


def route_pairs(GT, o, d, want_paths=False):
    """Shortest-path cost o[i] -> d[i] for all i (inf if unreachable)."""
    o = np.asarray(o, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    out = np.full(len(o), np.inf)
    paths = [None] * len(o) if want_paths else None
    if len(o) == 0:
        return out, paths
    tg, inv = np.unique(d, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(tg))]
    dl, pl = dist_to_targets(GT, tg, [o[g] for g in groups], want_paths)
    for k, g in enumerate(groups):
        out[g] = dl[k]
        if want_paths:
            for j, i in enumerate(g):
                paths[i] = pl[k][j]
    return out, paths


class Simulator:
    """Baseline + closure scenarios for a fixed set of OD pairs."""

    def __init__(self, core, access, cost_mode="length", baseline=None):
        """
        access: dict layer -> (o_nodes, d_nodes) arrays (one per pair)
        baseline: optional dict from Simulator.baseline_state() to skip the
                  baseline computation in worker processes.
        """
        self.core = core
        self.cost_mode = cost_mode
        self.layers = {m: Layer(core, m, cost_mode) for m in LAYERS}
        self.access = {m: (np.asarray(access[m][0], np.int64),
                           np.asarray(access[m][1], np.int64))
                       for m in LAYERS}
        self.npairs = len(self.access["car"][0])
        if baseline is None:
            self._compute_baseline()
        else:
            self.dist0 = baseline["dist0"]
            self.pk = baseline["pk"]
            self.pp = baseline["pp"]

    def _compute_baseline(self):
        self.dist0, self.pk, self.pp = {}, {}, {}
        for m in LAYERS:
            lay = self.layers[m]
            _, GT = lay.graphs(None)
            o, d = self.access[m]
            dist, paths = route_pairs(GT, o, d, want_paths=True)
            self.dist0[m] = dist
            keys = [p if p is not None else np.zeros(0, np.int64)
                    for p in paths]
            self.pk[m] = (np.concatenate(keys) if keys
                          else np.zeros(0, np.int64))
            self.pp[m] = np.repeat(np.arange(self.npairs),
                                   [len(k) for k in keys])

    def baseline_state(self):
        return {"dist0": self.dist0, "pk": self.pk, "pp": self.pp}

    def run(self, closed, frag=True):
        """
        closed: boolean mask over all city edges.
        Returns (rows, frag_rows):
          rows: list of (layer, pair, new_dist, iso_flag, inc_flag) for
                every pair that was re-routed or touches the closure
          frag_rows: list of dicts per layer
        """
        rows, frows = [], []
        for m in LAYERS:
            lay = self.layers[m]
            G, GT = lay.graphs(closed)
            ckeys = lay.closed_keys(closed)
            o, d = self.access[m]
            base = self.dist0[m]
            # closure nodes of this layer
            ce = closed[lay.e]
            inc_nodes = np.zeros(self.core.n, dtype=bool)
            inc_nodes[lay.eu[ce]] = True
            inc_nodes[lay.ev[ce]] = True
            deg = lay.degree_after(closed)
            iso_nodes = lay.nodes & (deg == 0)
            iso = iso_nodes[o] | iso_nodes[d]
            inc = inc_nodes[o] | inc_nodes[d]
            if len(ckeys):
                hit = np.isin(self.pk[m], ckeys, assume_unique=False)
                aff = np.unique(self.pp[m][hit])
            else:
                aff = np.zeros(0, dtype=np.int64)
            aff = aff[np.isfinite(base[aff])]
            new = base.copy()
            if len(aff):
                nd, _ = route_pairs(GT, o[aff], d[aff])
                new[aff] = nd
            report = np.zeros(self.npairs, dtype=bool)
            report[aff] = True
            report |= iso | inc
            for i in np.where(report)[0]:
                rows.append((m, int(i), float(new[i]), bool(iso[i]),
                             bool(inc[i])))
            if frag:
                lab = connected_components(G, directed=True,
                                           connection="weak")[1]
                nodes = lay.nodes
                cnt = np.bincount(lab[nodes])
                lcc1 = (lab == int(np.argmax(cnt))) & nodes
                cut = lay.lcc0 & ~lcc1
                pop = self.core.pop
                frows.append({
                    "layer": m,
                    "lcc0_n": int(lay.lcc0.sum()),
                    "lcc0_pop": float(pop[lay.lcc0].sum()),
                    "lcc1_n": int(lcc1.sum()),
                    "cut_n": int(cut.sum()),
                    "cut_pop": float(pop[cut].sum()),
                    "cut_xiso_n": int((cut & ~iso_nodes).sum()),
                    "cut_xiso_pop": float(pop[cut & ~iso_nodes].sum()),
                    "cut_xinc_n": int((cut & ~inc_nodes).sum()),
                    "cut_xinc_pop": float(pop[cut & ~inc_nodes].sum()),
                    "iso_n": int(iso_nodes.sum()),
                    "inc_n": int((inc_nodes & nodes).sum()),
                    "closed_layer_edges": int(ce.sum()),
                    "rerouted": int(len(aff)),
                })
        return rows, frows
