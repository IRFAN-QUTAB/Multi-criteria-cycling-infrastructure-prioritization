

import pickle

import numpy as np

from .network import Core, Simulator

_SIM = None
_PAIR_IDS = None


def init_worker(core_path, access_path, base_path, cost_mode):
    global _SIM, _PAIR_IDS
    core = Core.load(core_path)
    z = np.load(access_path)
    access = {m: (z["o_" + m], z["d_" + m])
              for m in ("car", "bike_all", "bike_safe")}
    _PAIR_IDS = z["pair_ids"]
    with open(base_path, "rb") as f:
        base = pickle.load(f)
    _SIM = Simulator(core, access, cost_mode, baseline=base)


def run_task(task):
    sid, closed_idx = task
    mask = np.zeros(_SIM.core.m, dtype=bool)
    mask[closed_idx] = True
    rows, frows = _SIM.run(mask)
    rows = [(sid, m, int(_PAIR_IDS[i]), d, iso, inc)
            for (m, i, d, iso, inc) in rows]
    for fr in frows:
        fr["scenario"] = sid
    return sid, rows, frows
