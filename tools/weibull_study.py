#!/usr/bin/env python3
"""weibull_study.py - how the Weibull distribution of cohesive strength and toughness controls cracking.

Design: Weibull modulus m x correlation length lc x seeds, single pellets, one heating ramp.
  m   : constant (no scatter), 4, 6, 8, 12, 20      (applied to strength AND toughness)
  lc  : 0 (independent per interface element, weakest-link scaled), 0.25, 0.5, 1.0 mm
  rho : toughness-strength correlation (default 0.5; --rho to change)
The mean strength (130 MPa) and toughness (4 J/m2) are held fixed, so only the spread and its spatial
structure change: the crack moves through regions of different toughness.

Metrics: onset dT (first fully cracked cohesive point), crack area at dT*, final crack area, dissipated
energy, fragments, radial planes cracked. Per (m, lc): mean, CV, bootstrap 95 % CI; Weibull fit of the
onset dT across seeds (its modulus vs the input m tells how much of the material scatter survives the
weakest-link selection by the stress field).

  python tools/weibull_study.py plan [--seeds 8 --profile lean]
  python tools/study_common.py run studies/weibull --parallel 2 --threads 4
  python tools/weibull_study.py analyze studies/weibull
  python tools/weibull_study.py plot studies/weibull
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_convergence as mc  # noqa: E402
import pczio  # noqa: E402
import study_common as sc  # noqa: E402

M_LEVELS = [0, 4, 6, 8, 12, 20]  # 0 = constant
LC_LEVELS = [0.0, 0.25e-3, 0.5e-3, 1.0e-3]
PROFILES = {"full": (M_LEVELS, LC_LEVELS, 10), "lean": ([0, 4, 8, 20], [0.0, 0.5e-3], 5)}


def plan(a):
    ms, lcs, seeds = PROFILES[a.profile]
    seeds = a.seeds or seeds
    sdir = os.path.join(a.root, a.name)
    hist = mc.heating(a.q_peak, a.t_ramp)
    cases, k = [], 0
    for m, lc in itertools.product(ms, lcs):
        if m == 0 and lc != lcs[0]:
            continue  # no scatter -> correlation length irrelevant
        for seed in range(1, (1 if m == 0 else seeds) + 1):
            ov = {**mc.level_mesh(a.nr), "loading.power_history_W_per_cm": sc.table(hist), "time.end": a.t_ramp,
                  "time.dt_max": 5.0, "weibull.seed": seed, "weibull.correlation_length": lc,
                  "weibull.toughness_strength_correlation": a.rho, "output.times": "",
                  "output.on_crack_area_growth": 0}
            if m == 0:
                ov.update({"strength.distribution": "constant", "toughness.distribution": "constant"})
            else:
                ov.update({"strength.weibull_modulus": m, "toughness.weibull_modulus": m})
                if lc == 0:
                    ov["weibull.reference_area"] = (2 * np.pi * mc.R0 / 32) * (mc.H0 / 24)
            cid = sc.case_id("wb", k, ov)
            cases.append(sc.Case(cid, ov, factors={"weibull_m": m, "lc_mm": lc * 1e3, "seed": seed}))
            k += 1
    sc.plan_study(sdir, a.name, cases, {"purpose": "Weibull modulus x correlation length", "profile": a.profile,
                                        "m_levels": ms, "lc_m": lcs, "seeds": seeds, "rho": a.rho, "nr": a.nr},
                  mc.BASE, threads=a.threads, card=__doc__.split("Metrics")[0])


def _boot_ci(v, n=2000, rng=np.random.default_rng(0)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _weibull_fit(v):
    """maximum-likelihood 2-parameter Weibull fit (modulus, scale)"""
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 3 or np.std(v) == 0:
        return np.nan, np.nan
    try:
        from scipy.stats import weibull_min
        c, loc, scale = weibull_min.fit(v, floc=0)
        return float(c), float(scale)
    except Exception:
        return np.nan, np.nan


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if pczio.run_complete(d):
            rows.append(dict(case_id=c.case_id, weibull_m=c.weibull_m, lc_mm=c.lc_mm, seed=c.seed, **mc.run_metrics(d)))
    df = pd.DataFrame(rows)
    dd = os.path.join(sdir, "derived")
    sc.save_table(df, os.path.join(dd, "weibull_runs.csv"), "One row per completed run", sources=list(df.get("case_id", [])))
    if df.empty:
        print("no completed runs yet")
        return
    agg = []
    for (m, lc), g in df.groupby(["weibull_m", "lc_mm"]):
        row = dict(weibull_m=m, lc_mm=lc, n=len(g))
        for col in ["onset_dT_K", "crack_area_final_mm2", "dissipated_final_mJ", "n_fragments_final",
                    "radial_planes_cracked"] + [f"A_dT{v}_mm2" for v in mc.DT_STAR]:
            v = g[col].astype(float)
            row[f"{col}_mean"] = v.mean()
            row[f"{col}_cv"] = v.std(ddof=1) / v.mean() if len(v) > 1 and v.mean() else np.nan
            lo, hi = _boot_ci(v)
            row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = lo, hi
        row["onset_weibull_modulus"], row["onset_weibull_scale_K"] = _weibull_fit(g.onset_dT_K)
        agg.append(row)
    agg = pd.DataFrame(agg)
    sc.save_table(agg, os.path.join(dd, "weibull_levels.csv"), "Statistics per (m, lc): mean, CV, bootstrap CI, onset Weibull fit")
    print(agg[["weibull_m", "lc_mm", "n", "onset_dT_K_mean", "onset_dT_K_cv", "crack_area_final_mm2_mean",
               "n_fragments_final_mean"]].to_string(index=False))


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    dd, fd = os.path.join(a.study_dir, "derived"), os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    agg = pd.read_csv(os.path.join(dd, "weibull_levels.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for lc, g in agg[agg.weibull_m > 0].groupby("lc_mm"):
        g = g.sort_values("weibull_m")
        lab = "per element" if lc == 0 else f"lc = {lc:g} mm"
        axs[0].errorbar(g.weibull_m, g.onset_dT_K_mean, yerr=[g.onset_dT_K_mean - g.onset_dT_K_ci_lo,
                                                              g.onset_dT_K_ci_hi - g.onset_dT_K_mean], marker="o", capsize=2, label=lab)
        axs[1].plot(g.weibull_m, g.onset_dT_K_cv, marker="o", label=lab)
        axs[2].plot(g.weibull_m, g.crack_area_final_mm2_mean, marker="o", label=lab)
    det = agg[agg.weibull_m == 0]
    if len(det):
        axs[0].axhline(det.onset_dT_K_mean.iloc[0], color="k", ls=":", lw=1, label="no scatter")
        axs[2].axhline(det.crack_area_final_mm2_mean.iloc[0], color="k", ls=":", lw=1, label="no scatter")
    m = np.linspace(3, 25, 50)
    from math import gamma
    axs[1].plot(m, [np.sqrt(gamma(1 + 2 / x) / gamma(1 + 1 / x) ** 2 - 1) for x in m], color="0.5", ls="--",
                label="input strength CV")
    axs[0].set_ylabel("onset ΔT [K]")
    axs[1].set_ylabel("CV")
    axs[2].set_ylabel("final crack area [mm²]")
    for ax in axs:
        ax.set_xlabel("Weibull modulus m")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(fd, "weibull_summary.png"), dpi=150)
    plt.close(fig)
    print("figures ->", fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--root", default="studies")
    p.add_argument("--name", default="weibull")
    p.add_argument("--profile", choices=list(PROFILES), default="lean")
    p.add_argument("--seeds", type=int, default=0)
    p.add_argument("--nr", type=int, default=12)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--q-peak", type=float, default=400.0)
    p.add_argument("--t-ramp", type=float, default=400.0)
    p.add_argument("--threads", type=int, default=0)
    for name in ("analyze", "plot"):
        q = sp.add_parser(name)
        q.add_argument("study_dir")
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
