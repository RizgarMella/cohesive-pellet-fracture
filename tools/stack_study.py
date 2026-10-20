#!/usr/bin/env python3
"""stack_study.py - sympathetic cracking between two stacked pellets (inputs/two_pellet_sympathetic.ini).

Pellet 0 (bottom) has a weak flaw at theta = 0 below its upper end face and cracks first; pellet 1 (top)
can crack on any radial element boundary. Both pellets are pressed (top pressure, preload step), heated
and held.

Factors
  pressure_MPa   nominal axial pressure on the top face          : 0, 2, 10, 30
  mu             interface friction coefficient (-1 = stick)      : 0, 0.2, 0.5, 1.0, -1
  contact_frac   nominal contact-area fraction (annulus, dished)  : 1.0, 0.75, 0.5
  middle         seeded (flaw in pellet 0) | control (no flaw: same loads, CONTROL for every combination)
  seed           Weibull realisation
Design 'full' = full factorial (seeded x 2 seeds + control x 1 seed); 'oat' = one factor at a time around
(10 MPa, 0.5, 0.75) plus the (pressure, mu) corners at each contact level.

Metrics (derived/stack_runs.csv)
  crack area of each pellet at peak power and at the end; sympathetic excess = pellet-1 crack area minus
  the matched CONTROL; crack ALIGNMENT across the interface from angular_*.csv (damage vs angle in the
  element layer next to each end face): Pearson r of the two profiles and enrichment =
  P(pellet-1 face crack within +-w of a pellet-0 crack mouth) / (fraction of circumference within +-w)
  [1 = independent, >1 = seeded]; interface state at the end: contact force, mean/max pressure, slip
  fraction, shear force.

  python tools/stack_study.py plan [--design oat]
  python tools/study_common.py run studies/stack --parallel 1 --threads 8
  python tools/stack_study.py analyze studies/stack
  python tools/stack_study.py plot studies/stack
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pczio  # noqa: E402
import study_common as sc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "inputs", "two_pellet_sympathetic.ini")
PRESSURE_MPA = [0.0, 2.0, 10.0, 30.0]
MU = [0.0, 0.2, 0.5, 1.0, -1.0]
CONTACT = [1.0, 0.75, 0.5]
SEEDS = [11, 12]


def design(kind: str):
    combos = []
    if kind == "full":
        for p, mu, cf in itertools.product(PRESSURE_MPA, MU, CONTACT):
            combos += [(p, mu, cf, "seeded", s) for s in SEEDS] + [(p, mu, cf, "control", SEEDS[0])]
    else:
        base = (10.0, 0.5, 0.75)
        pts = {base}
        pts |= {(p, base[1], base[2]) for p in PRESSURE_MPA} | {(base[0], m, base[2]) for m in MU}
        pts |= {(base[0], base[1], c) for c in CONTACT}
        pts |= {(p, m, c) for c in CONTACT for p in (PRESSURE_MPA[0], PRESSURE_MPA[-1]) for m in (MU[0], MU[3])}
        for p, mu, cf in sorted(pts):
            combos += [(p, mu, cf, "seeded", SEEDS[0]), (p, mu, cf, "control", SEEDS[0])]
    return combos


def plan(a):
    sdir = os.path.join(a.root, a.name)
    cases = []
    for k, (p, mu, cf, mid, seed) in enumerate(design(a.design)):
        ov = {"boundary.top_pressure": sc.table([(0, 0), (10, p * 1e6), (a.t_end, p * 1e6)]),
              "interface.friction": "stick" if mu < 0 else mu, "interface.contact_fraction": cf,
              "weibull.seed": seed, "time.end": a.t_end, "output.times": f"10,{a.t_peak:g},{a.t_end:g}"}
        if mid == "control":
            ov["weibull.flaws"] = "none"
        cid = sc.case_id("st", k, ov)
        cases.append(sc.Case(cid, ov, factors={"pressure_MPa": p, "mu": mu, "contact_frac": cf, "middle": mid,
                                               "seed": seed}))
    sc.plan_study(sdir, a.name, cases, {"purpose": "sympathetic cracking between stacked pellets", "design": a.design,
                                        "factors": {"pressure_MPa": PRESSURE_MPA, "mu": MU, "contact_frac": CONTACT}},
                  BASE, threads=a.threads, card=__doc__.split("Metrics")[0])


def alignment(face0, face1, w_bins=2, thr=0.5):
    """Pearson r of two angular damage profiles + enrichment of face-1 cracks near face-0 crack mouths."""
    a, b = np.asarray(face0, float), np.asarray(face1, float)
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    ma, mb = a > thr, b > thr
    if ma.sum() == 0 or mb.sum() == 0:
        return r, np.nan, int(mb.sum())
    near = np.zeros_like(ma)
    for s in range(-w_bins, w_bins + 1):
        near |= np.roll(ma, s)
    expected = near.mean()
    observed = (mb & near).sum() / mb.sum()
    return r, float(observed / expected) if expected > 0 else np.nan, int(mb.sum())


def _angular_at(d, t):
    import pandas as pd
    p = os.path.join(d, "outputs.csv")
    if not os.path.exists(p):
        return None
    o = pd.read_csv(p)
    i = int(o.iloc[int(np.argmin(np.abs(o.time_s - t)))]["index"])
    f = os.path.join(d, f"angular_{i:04d}.csv")
    return pczio.read_angular(f) if os.path.exists(f) else None


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    sdef = pczio.read_json(os.path.join(sdir, "study.json"))
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if not pczio.run_complete(d):
            continue
        r = pczio.read_run(d)
        h, s = r["history"], r["summary"]
        row = dict(case_id=c.case_id, pressure_MPa=c.pressure_MPa, mu=c.mu, contact_frac=c.contact_frac,
                   middle=c.middle, seed=c.seed)
        t_peak = a.t_peak
        hp = h.iloc[int(np.argmin(np.abs(h.time_s - t_peak)))]
        he = h.iloc[-1]
        for p in (0, 1):
            row[f"A_p{p}_peak_mm2"] = hp[f"crack_area_p{p}_mm2"]
            row[f"A_p{p}_final_mm2"] = he[f"crack_area_p{p}_mm2"]
        for col in ("if_contact_force_N", "if_mean_pressure_MPa", "if_max_pressure_MPa", "if_slip_fraction",
                    "if_shear_force_N", "pin_reaction_N"):
            row[col] = he[col]
        for tag, t in (("peak", t_peak), ("final", he.time_s)):
            ang = _angular_at(d, t)
            if ang is None:
                continue
            rr, enr, n = alignment(ang["p0_top_damage"], ang["p1_bottom_damage"])
            row[f"pearson_{tag}"], row[f"enrichment_{tag}"], row[f"p1_face_cracks_{tag}"] = rr, enr, n
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no completed runs yet")
        return
    ctrl = df[df.middle == "control"].set_index(["pressure_MPa", "mu", "contact_frac"])
    for tag in ("peak", "final"):
        def excess(r, tag=tag):
            k = (r.pressure_MPa, r.mu, r.contact_frac)
            if r.middle != "seeded" or k not in ctrl.index:
                return np.nan
            return r[f"A_p1_{tag}_mm2"] - np.atleast_1d(ctrl.loc[k, f"A_p1_{tag}_mm2"])[0]
        df[f"sympathetic_excess_{tag}_mm2"] = df.apply(excess, axis=1)
    dd = os.path.join(sdir, "derived")
    sc.save_table(df, os.path.join(dd, "stack_runs.csv"),
                  "Per run: crack area per pellet (peak/final), sympathetic excess vs matched control, "
                  "cross-interface alignment, interface state", sources=list(df.case_id))
    eff = (df[df.middle == "seeded"].groupby(["pressure_MPa", "mu", "contact_frac"])
           [[c for c in df.columns if c.startswith(("sympathetic", "enrichment", "pearson", "if_"))]].mean().reset_index())
    sc.save_table(eff, os.path.join(dd, "stack_effects.csv"), "Seed means per factor combination (seeded runs)")
    print(eff.to_string(index=False))


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    dd, fd = os.path.join(a.study_dir, "derived"), os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    eff = pd.read_csv(os.path.join(dd, "stack_effects.csv"))
    for metric, cmap, center, label in (("enrichment_final", "RdBu_r", 1.0, "alignment enrichment (1 = none)"),
                                        ("sympathetic_excess_final_mm2", "RdBu_r", 0.0, "sympathetic excess [mm²]")):
        if metric not in eff:
            continue
        cfs = sorted(eff.contact_frac.unique())
        fig, axs = plt.subplots(1, len(cfs), figsize=(4 * len(cfs), 3.6), squeeze=False, constrained_layout=True)
        vmax = np.nanmax(np.abs(eff[metric] - center)) or 1
        for ax, cf in zip(axs[0], cfs):
            g = eff[eff.contact_frac == cf].pivot_table(index="mu", columns="pressure_MPa", values=metric)
            im = ax.imshow(g.values, cmap=cmap, vmin=center - vmax, vmax=center + vmax, origin="lower", aspect="auto")
            ax.set_xticks(range(len(g.columns)), [f"{v:g}" for v in g.columns])
            ax.set_yticks(range(len(g.index)), ["stick" if v < 0 else f"{v:g}" for v in g.index])
            ax.set_xlabel("pressure [MPa]")
            ax.set_ylabel("friction μ")
            ax.set_title(f"contact fraction {cf:g}")
        fig.colorbar(im, ax=axs[0].tolist(), label=label)
        fig.savefig(os.path.join(fd, f"stack_{metric}.png"), dpi=150)
        plt.close(fig)
    print("figures ->", fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--root", default="studies")
    p.add_argument("--name", default="stack")
    p.add_argument("--design", choices=["oat", "full"], default="oat")
    p.add_argument("--t-peak", type=float, default=310.0)
    p.add_argument("--t-end", type=float, default=400.0)
    p.add_argument("--threads", type=int, default=0)
    for name in ("analyze", "plot"):
        q = sp.add_parser(name)
        q.add_argument("study_dir")
        q.add_argument("--t-peak", type=float, default=310.0)
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
