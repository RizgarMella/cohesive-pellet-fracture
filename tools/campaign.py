#!/usr/bin/env python3
"""campaign.py - parametric campaign feeding the crack/fragment metamodel (features.py -> fit_pysr.py).

Scrambled Sobol design over geometry, material distributions and power histories (single pellets).
The variables and their ASSUMED distributions are in DISTS - replace them with measured ones.
Stage 1 (lean) = 128 train + 32 validation runs; `plan --extend` adds the next 128 + 32 from the same Sobol
sequence (only worth it if `fit_pysr.py --learning-curve` says the last doubling still helped).

Histories: ramp to q_peak in t_ramp, hold, then optionally a second ramp of +ramp2 x q_peak over 30 min
and a 12 h hold (the paper's power-ramp test). The mesh comes from studies/meshconv/derived/convergence.json
when it exists (else nr = 12).

  python tools/campaign.py plan [--n-train 128 --n-val 32]
  python tools/study_common.py run studies/campaign-A --parallel 2 --threads 4   (or MPI / Slurm)
  python tools/campaign.py collect studies/campaign-A          # -> derived/dataset.csv
  python tools/fit_pysr.py studies/campaign-A
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as ft  # noqa: E402
import mesh_convergence as mc  # noqa: E402
import pczio  # noqa: E402
import study_common as sc  # noqa: E402

# (kind, parameters): U = uniform(lo, hi), N = normal(mu, sd), LN = lognormal(median, sigma_log), LU = log-uniform
DISTS = {
    "R_m": ("U", 3.5e-3, 5.0e-3),
    "HD": ("U", 0.8, 1.6),
    "E_Pa": ("N", 219e9, 8e9),
    "nu": ("U", 0.30, 0.35),
    "alpha": ("N", 10.0e-6, 0.5e-6),
    "k_scale": ("U", 0.8, 1.2),
    "sigma_Pa": ("LN", 130e6, 0.2),
    "Gc_Jpm2": ("LN", 4.0, 0.3),
    "weibull_m": ("U", 4.0, 16.0),
    "lc_m": ("U", 0.2e-3, 1.0e-3),
    "rho": ("U", 0.0, 1.0),
    "q_peak_Wpcm": ("U", 150.0, 450.0),
    "t_ramp_s": ("LU", 60.0, 3600.0),
    "hold_s": ("LU", 1e2, 3e7),
    "ramp2": ("U", 0.0, 0.6),
}


def _inv(u, kind, a, b):
    from scipy.stats import norm
    u = min(max(u, 1e-9), 1 - 1e-9)
    if kind == "U":
        return a + u * (b - a)
    if kind == "N":
        return a + b * norm.ppf(u)
    if kind == "LN":
        return a * math.exp(b * norm.ppf(u))
    if kind == "LU":
        return math.exp(math.log(a) + u * (math.log(b) - math.log(a)))
    raise ValueError(kind)


def sample(n: int, seed: int, skip: int = 0):
    from scipy.stats import qmc
    eng = qmc.Sobol(d=len(DISTS), scramble=True, seed=seed)
    if skip:
        eng.fast_forward(skip)
    U = eng.random(n)
    return [{k: _inv(u[i], *DISTS[k]) for i, k in enumerate(DISTS)} for u in U]


def make_history(s: dict):
    q, tr, hold = s["q_peak_Wpcm"], s["t_ramp_s"], s["hold_s"]
    rows = [(0.0, 0.0), (tr, q), (tr + hold, q)]
    if s["ramp2"] > 0.02:
        rows += [(tr + hold + 1800.0, q * (1 + s["ramp2"])), (tr + hold + 1800.0 + 43200.0, q * (1 + s["ramp2"]))]
    return rows


def mesh_for(R, H, conv):
    nr = conv.get("mesh.nr", 12) if conv else 12
    m = mc.level_mesh(int(nr))
    m["mesh.nz"] = 8 * max(1, round(m["mesh.nz"] * (H / mc.H0) * (mc.R0 / R) / 8))
    return m


def to_overrides(s: dict, conv: dict | None, seed: int) -> dict:
    H = s["HD"] * 2 * s["R_m"]
    hist = make_history(s)
    return {"geometry.radius": s["R_m"], "geometry.height": H, **mesh_for(s["R_m"], H, conv),
            "material.youngs_modulus": s["E_Pa"], "material.poisson_ratio": s["nu"],
            "material.thermal_expansion": s["alpha"], "material.conductivity_scale": s["k_scale"],
            "strength.mean": s["sigma_Pa"], "strength.weibull_modulus": s["weibull_m"],
            "toughness.mean": s["Gc_Jpm2"], "toughness.weibull_modulus": s["weibull_m"],
            "weibull.correlation_length": s["lc_m"], "weibull.toughness_strength_correlation": s["rho"],
            "weibull.seed": seed, "loading.power_history_W_per_cm": sc.table(hist), "time.end": hist[-1][0],
            "time.dt_max": max(5.0, 0.05 * s["hold_s"]), "output.times": "", "output.on_crack_area_growth": 0}


def plan(a):
    sdir = os.path.join(a.root, a.name)
    conv_p = os.path.join(a.root, "meshconv", "derived", "convergence.json")
    conv = pczio.read_json(conv_p) if os.path.exists(conv_p) else None
    skip = 0
    old = []
    if a.extend:
        import pandas as pd
        old_cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
        skip = int(pczio.read_json(os.path.join(sdir, "study.json")).get("sobol_used", 0))
        old = old_cases.to_dict("records")
    tr = sample(a.n_train, a.seed, skip)
    va = sample(a.n_val, a.seed + 1, skip)
    cases = []
    k = skip
    for split, ss in (("train", tr), ("val", va)):
        for s in ss:
            ov = to_overrides(s, conv, seed=k + 1)
            cid = sc.case_id("cp", k, ov)
            cases.append(sc.Case(cid, ov, factors={kk: float(v) for kk, v in s.items()}, split=split))
            k += 1
    if old:
        for r in old:  # keep earlier cases (re-planned identically from their overrides)
            ov = dict(x.split("=", 1) for x in str(r["overrides"]).split())
            cases.insert(0, sc.Case(r["case_id"], ov, factors={kk: r[kk] for kk in DISTS}, split=r["split"]))
    sc.plan_study(sdir, a.name, cases, {"purpose": "metamodel campaign", "dists": DISTS, "seed": a.seed,
                                        "sobol_used": skip + max(a.n_train, a.n_val), "mesh": conv},
                  mc.BASE, threads=a.threads, card=__doc__.split("Histories")[0])


def collect(a):
    import pandas as pd
    sdir = a.study_dir
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        s = {k: c[k] for k in DISTS}
        H = s["HD"] * 2 * s["R_m"]
        geom = dict(R=s["R_m"], H=H)
        mat = dict(E=s["E_Pa"], nu=s["nu"], alpha=s["alpha"], k_scale=s["k_scale"], sigma=s["sigma_Pa"],
                   Gc=s["Gc_Jpm2"], weibull_m=s["weibull_m"], lc=s["lc_m"], rho=s["rho"])
        row = dict(case_id=c.case_id, split=c.split, **ft.groups(geom, mat, make_history(s)))
        if pczio.run_complete(d):
            row.update(status="complete", **ft.targets(pczio.read_json(os.path.join(d, "summary.json")), geom, mat))
        else:
            row["status"] = "missing"
        rows.append(row)
    df = pd.DataFrame(rows)
    sc.save_table(df, os.path.join(sdir, "derived", "dataset.csv"),
                  "Features (dimensionless groups) and targets per case", sources=list(df.case_id))
    print(df.status.value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--root", default="studies")
    p.add_argument("--name", default="campaign-A")
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--n-val", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--extend", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    q = sp.add_parser("collect")
    q.add_argument("study_dir")
    a = ap.parse_args()
    {"plan": plan, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
