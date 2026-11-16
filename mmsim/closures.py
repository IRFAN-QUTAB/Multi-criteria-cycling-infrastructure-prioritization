"""Closure scenarios as boolean masks over the city edges."""

import numpy as np


def alternatives(city, top_pct=50):
    """Top top_pct% named roads by dual PageRank (same rule as before:
    n = int(total * top_pct / 100)), ties broken by name. Roads without
    any ROUTE edge of that name are dropped and reported."""
    items = sorted(city.road_pagerank.items(), key=lambda t: (-t[1], t[0]))
    n = max(1, int(len(items) * top_pct / 100))
    top = items[:n]
    keep = [(r, pr) for r, pr in top if r in city.road_edges]
    missing = [r for r, _ in top if r not in city.road_edges]
    return keep, missing


def all_named_roads(city):
    return sorted(r for r in city.road_node if r in city.road_edges)


def single_closure(city, road, depth=0.25, min_seg=10, legacy=False):
    """
    Close one road.
    New rule: 25% of the road's PHYSICAL segments (both directions), at
    least min_seg segments, chosen at the junctions with the highest car
    betweenness; roads with at most min_seg segments (or depth >= 1) are
    closed entirely.
    legacy=True reproduces the Neo4j version (directed edges counted and
    selected one by one).
    """
    E = city.road_edges[road]
    bc = city.car_bc
    score = bc[city.u[E]] + bc[city.v[E]]
    if legacy:
        n_all = len(E)
        n_close = min(max(min_seg, int(n_all * depth)), n_all)
        if n_all <= n_close:
            sel, ctype = E, "full_road"
        else:
            order = np.lexsort((city.edge_id[E], -score))
            sel, ctype = E[order[:n_close]], "partial"
    else:
        segs = city.seg[E]
        useg = np.unique(segs)
        n_phys = len(useg)
        n_close = min(max(min_seg, int(n_phys * depth)), n_phys)
        if depth >= 1.0 or n_phys <= n_close:
            sel, ctype = E, "full_road"
        else:
            best = {}
            for s, sc in zip(segs, score):
                if s not in best or sc > best[s]:
                    best[s] = sc
            ss = np.array(sorted(best, key=lambda s: (-best[s], s)))
            chosen = ss[:n_close]
            sel, ctype = E[np.isin(segs, chosen)], "partial"
    mask = np.zeros(city.m, dtype=bool)
    mask[sel] = True
    return mask, ctype, int(len(sel)), int(len(np.unique(city.seg[sel])))


def roads_closure(city, roads):
    E = city.edges_of_roads(roads)
    mask = np.zeros(city.m, dtype=bool)
    mask[E] = True
    return mask, int(len(E)), int(len(np.unique(city.seg[E])))
