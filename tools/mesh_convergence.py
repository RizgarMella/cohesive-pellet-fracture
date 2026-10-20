#!/usr/bin/env python3
"""mesh_convergence.py - discretisation study based on CRACK SURFACE AREA.

Families (each refined over LEVELS, several Weibull realisations per level):
  field     : Weibull strength/toughness from a correlated random field of FIXED physical correlation
              length (weibull.correlation_length). Refining the mesh samples the same material, so the
              crack area should converge -> the main convergence test.
  element   : independent Weibull draw per interface element with weakest-link area scaling
              (weibull.correlation_length = 0, weibull.reference_area = facet area at the base level).
              Shows how per-element scatter behaves under refinement.
  constant  : no scatter (deterministic). With uniform loading every symmetric plane initiates together,
              the classic difficulty noted by Williamson & Knoll; included for reference (1 seed).
Each level scales (nr, ntheta, nz) together; ntheta stays a multiple of the radial-plane count and nz of
the slab count, so the potential crack planes are the same at every level.

Metrics (history.csv / summary.json):
  crack area A(dT*) at centre-surface temperature differences dT* (first heating ramp), onset dT,
  final crack area, dissipated energy, fragment count, number of radial planes cracked, and the mesh
  parameter zeta = l_cz / h (Turon et al. 2007: ~3 needed to resolve the cohesive zone; the paper
  found engineering accuracy from zeta ~ 0.7).
Analysis per (family, metric): seed mean / sd / sem per level; Celik et al. (2008) GCI on the three
finest levels (iterative order p for non-uniform refinement ratios), Richardson extrapolation, and a
seed-noise test (|phi1 - phi2| <= 2 x combined sem). The coarsest level within --tol of the
extrapolated A(dT_ref) is recorded in derived/convergence.json.

  python tools/mesh_convergence.py plan    [--root studies --seeds 4 --profile lean]
  python tools/study_common.py run studies/meshconv --parallel 1 --threads 8     (or MPI / Slurm)
  python tools/mesh_convergence.py analyze studies/meshconv [--tol 0.1 --dt-ref 300]
  python tools/mesh_convergence.py plot    studies/meshconv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pczio  # noqa: E402
import study_common as sc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "inputs", "single_pellet_demo.ini")
# radial elements per level; ntheta and nz follow
NR_LEVELS = [6, 9, 12, 18, 24, 36]
PROFILES = {"full": {"field": NR_LEVELS, "element": NR_LEVELS, "constant": NR_LEVELS},
            "lean": {"field": [6, 9, 12, 18, 24], "element": [6, 12, 24], "constant": [6, 12, 24]}}
DT_STAR = [100, 150, 200, 300, 400]
R0, H0 = 4.67e-3, 14e-3
N_RADIAL, N_SLABS = 8, 8


def level_mesh(nr: int) -> dict:
    nth = N_RADIAL * max(1, math.ceil(nr * 32 / 12 / N_RADIAL))
    nz = N_SLABS * max(1, round(nr * 2 / N_SLABS))
    return {"mesh.nr": nr, "mesh.ntheta": nth, "mesh.nz": nz}


def heating(q_peak=400.0, t_ramp=400.0):
    """single heating ramp (W/cm) - no burnup effects matter on this time scale"""
    return [(0, 0), (t_ramp, q_peak)]


def plan(a):
    sdir = os.path.join(a.root, a.name)
    prof = dict(PROFILES[a.profile])
    if a.levels:
        prof = {f: list(a.levels) for f in prof}
    if a.families:
        prof = {f: v for f, v in prof.items() if f in a.families}
    cases, k = [], 0
    hist = heating(a.q_peak, a.t_ramp)
    base_area = (2 * math.pi * R0 / 32) * (H0 / 24)  # facet area of the nr=12 base mesh (radial face ~ dr*dz)
    for fam, levels in prof.items():
        seeds = 1 if fam == "constant" else a.seeds
        for nr in levels:
            for seed in range(1, seeds + 1):
                ov = {**level_mesh(nr), "loading.power_history_W_per_cm": sc.table(hist), "time.end": a.t_ramp,
                      "time.dt_max": 5.0, "weibull.seed": seed, "output.times": "", "output.on_crack_area_growth": 0}
                if fam == "element":
                    ov.update({"weibull.correlation_length": 0.0, "weibull.reference_area": base_area})
                elif fam == "constant":
                    ov.update({"strength.distribution": "constant", "toughness.distribution": "constant"})
                h = R0 / nr
                cid = sc.case_id("mc", k, ov)
                cases.append(sc.Case(cid, ov, factors={"family": fam, "nr": nr, "seed": seed, "h_mm": h * 1e3,
                                                       "ntheta": ov["mesh.ntheta"], "nz": ov["mesh.nz"]}))
                k += 1
    sc.plan_study(sdir, a.name, cases, {"purpose": "mesh convergence on crack area", "profile": a.profile,
                                        "levels": prof, "dT_star_K": DT_STAR, "seeds": a.seeds,
                                        "heating_W_per_cm": hist},
                  BASE, threads=a.threads, card=__doc__.split("Analysis")[0])


def celik_gci(h, phi, fs=1.25):
    """Celik et al. (2008). h = (h1 < h2 < h3) fine -> coarse, phi likewise."""
    h1, h2, h3 = h
    f1, f2, f3 = phi
    out = dict(phi1=f1, phi2=f2, phi3=f3, r21=h2 / h1, r32=h3 / h2, p=np.nan, phi_ext=np.nan, ea21=np.nan,
               gci21=np.nan, kind="undetermined")
    e21, e32 = f2 - f1, f3 - f2
    if not np.all(np.isfinite([f1, f2, f3])) or e21 == 0 or e32 == 0 or f1 == 0:
        return out
    r21, r32 = h2 / h1, h3 / h2
    s = np.sign(e32 / e21)
    p = abs(np.log(abs(e32 / e21))) / np.log(r21)
    for _ in range(200):
        num, den = r21 ** p - s, r32 ** p - s
        if num <= 0 or den <= 0:
            return {**out, "kind": "oscillatory-diverging"}
        pn = abs(np.log(abs(e32 / e21)) + np.log(num / den)) / np.log(r21)
        if abs(pn - p) < 1e-10:
            break
        p = pn
    fext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea = abs((f1 - f2) / f1)
    out.update(p=p, phi_ext=fext, ea21=ea, gci21=fs * ea / (r21 ** p - 1), kind="monotone" if s > 0 else "oscillatory")
    return out


def run_metrics(d: str) -> dict:
    r = pczio.read_run(d)
    h, s, m = r["history"], r["summary"], r["manifest"]
    row = dict(onset_dT_K=s.get("onset_dT_K"), crack_area_final_mm2=sum(s.get("crack_area_final_mm2") or [np.nan]),
               dissipated_final_mJ=s.get("final_dissipated_energy_mJ"), n_fragments_final=s.get("final_n_fragments"),
               wall_s=s.get("wall_s"), steps=s.get("steps"),
               zeta=m.get("derived", {}).get("zeta_lcz_over_h"))
    planes = s.get("crack_area_by_plane_mm2") or []
    row["radial_planes_cracked"] = sum(1 for p in planes if p["kind"] == "radial" and p["area_mm2"] > 0.5)
    for v, a, e in zip(DT_STAR, pczio.value_at(h, "dT_K", "crack_area_mm2", DT_STAR),
                       pczio.value_at(h, "dT_K", "dissipated_energy_mJ", DT_STAR)):
        row[f"A_dT{v}_mm2"] = a
        row[f"E_dT{v}_mJ"] = e
    return row


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    per = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if not pczio.run_complete(d):
            continue
        per.append(dict(case_id=c.case_id, family=c.family, nr=c.nr, seed=c.seed, h_mm=c.h_mm, **run_metrics(d)))
    per = pd.DataFrame(per)
    dd = os.path.join(sdir, "derived")
    sc.save_table(per, os.path.join(dd, "meshconv_runs.csv"), "One row per completed run: metrics at dT* and final",
                  sources=list(per.case_id) if len(per) else [])
    if per.empty:
        print("no completed runs yet")
        return
    metrics = [c for c in per.columns if c.startswith(("A_dT", "E_dT"))] + [
        "onset_dT_K", "crack_area_final_mm2", "dissipated_final_mJ", "n_fragments_final", "radial_planes_cracked"]
    agg = []
    for (fam, nr), g in per.groupby(["family", "nr"]):
        row = dict(family=fam, nr=nr, h_mm=g.h_mm.iloc[0], zeta=g.zeta.iloc[0], n_seeds=len(g))
        for mname in metrics:
            v = g[mname].astype(float)
            row[f"{mname}_mean"] = v.mean()
            row[f"{mname}_sd"] = v.std(ddof=1) if v.count() > 1 else 0.0
            row[f"{mname}_sem"] = row[f"{mname}_sd"] / np.sqrt(max(1, v.count()))
        agg.append(row)
    agg = pd.DataFrame(agg)
    sc.save_table(agg, os.path.join(dd, "meshconv_levels.csv"), "Seed statistics per (family, level)")
    gci_rows = []
    for fam, g in agg.groupby("family"):
        g = g.sort_values("h_mm")
        if len(g) < 3:
            continue
        f3 = g.iloc[:3]
        for mname in metrics:
            res = celik_gci(tuple(f3.h_mm), tuple(f3[f"{mname}_mean"]))
            sem12 = float(np.hypot(f3[f"{mname}_sem"].iloc[0], f3[f"{mname}_sem"].iloc[1]))
            res.update(family=fam, metric=mname, h1_mm=f3.h_mm.iloc[0],
                       within_seed_noise=bool(abs(res["phi1"] - res["phi2"]) <= 2 * sem12) if sem12 > 0 else None)
            gci_rows.append(res)
    gci = pd.DataFrame(gci_rows)
    sc.save_table(gci, os.path.join(dd, "meshconv_gci.csv"),
                  "Celik GCI on the three finest levels per (family, metric); phi_ext = Richardson extrapolation")
    key = f"A_dT{int(a.dt_ref)}_mm2"
    rec = {"tol": a.tol, "metric": key, "candidates": []}
    for fam, g in agg.groupby("family"):
        g = g.sort_values("h_mm")
        gr = gci[(gci.family == fam) & (gci.metric == key)] if len(gci) else gci
        ref = gr.phi_ext.iloc[0] if len(gr) and np.isfinite(gr.phi_ext.iloc[0]) else g[f"{key}_mean"].iloc[0]
        g = g.assign(rel_err=(g[f"{key}_mean"] - ref).abs() / abs(ref) if ref else np.nan)
        ok = g[g.rel_err <= a.tol]
        pick = ok.sort_values("h_mm", ascending=False).iloc[0] if len(ok) else g.iloc[0]
        rec["candidates"].append(dict(family=fam, nr=int(pick.nr), h_mm=float(pick.h_mm), zeta=float(pick.zeta),
                                      rel_err=float(pick.rel_err), reference=float(ref),
                                      gci_kind=(gr.kind.iloc[0] if len(gr) else None)))
    best = next((c for c in rec["candidates"] if c["family"] == "field"), None)
    if best:
        rec.update(level_mesh(best["nr"]))
    with open(os.path.join(dd, "convergence.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    dd, fd = os.path.join(a.study_dir, "derived"), os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    agg = pd.read_csv(os.path.join(dd, "meshconv_levels.csv"))
    for fam, g in agg.groupby("family"):
        fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
        g = g.sort_values("h_mm", ascending=False)
        cmap = plt.get_cmap("Blues")
        for i, (_, r) in enumerate(g.iterrows()):
            mean = [r[f"A_dT{v}_mm2_mean"] for v in DT_STAR]
            sd = [r[f"A_dT{v}_mm2_sd"] for v in DT_STAR]
            col = cmap(0.35 + 0.6 * i / max(1, len(g) - 1))
            ax.plot(DT_STAR, mean, color=col, lw=2, label=f"h={r.h_mm:.3f} mm (ζ={r.zeta:.2f})")
            ax.fill_between(DT_STAR, np.subtract(mean, sd), np.add(mean, sd), color=col, alpha=0.15, lw=0)
        ax.set_xlabel("centre-surface ΔT [K]")
        ax.set_ylabel("crack area [mm²]")
        ax.set_title(f"{fam} family")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(fd, f"meshconv_{fam}_area_vs_dT.png"), dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    key = f"A_dT{int(a.dt_ref)}_mm2_mean"
    for fam, g in agg.groupby("family"):
        g = g.sort_values("h_mm")
        ax.errorbar(g.h_mm, g[key], yerr=g[key.replace("_mean", "_sem")], marker="o", ms=5, lw=1.5, capsize=2, label=fam)
    ax.set_xscale("log")
    ax.set_xlabel("element size h = R/nr [mm]")
    ax.set_ylabel(f"crack area at ΔT = {a.dt_ref:.0f} K [mm²]")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(fd, "meshconv_area_vs_h.png"), dpi=150)
    plt.close(fig)
    print("figures ->", fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--root", default="studies")
    p.add_argument("--name", default="meshconv")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--profile", choices=list(PROFILES), default="lean")
    p.add_argument("--q-peak", type=float, default=400.0, help="W/cm at the end of the ramp")
    p.add_argument("--t-ramp", type=float, default=400.0, help="s")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--levels", type=int, nargs="*", help="override the nr levels")
    p.add_argument("--families", nargs="*", choices=["field", "element", "constant"])
    for name in ("analyze", "plot"):
        q = sp.add_parser(name)
        q.add_argument("study_dir")
        q.add_argument("--tol", type=float, default=0.10)
        q.add_argument("--dt-ref", type=float, default=300.0)
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
