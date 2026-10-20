#!/usr/bin/env python3
"""render.py - pictures of a pelletcz run (matplotlib only; ParaView reads the same .vtu/.pvd files).

    python tools/render.py runs/single_pellet_demo                      # all figures -> <run>/figures
    python tools/render.py runs/X --frames 3,7,-1 --warp 150 --out docs/images

Figures
  cracking_3d.png  : per frame, the deformed pellet (warped) coloured by max principal stress, and the
                     internal crack surfaces (damage >= threshold) coloured by crack-face opening
  midplane.png     : hoop stress on the mid-height cross-section with the crack traces
  history.png      : temperature difference, crack area by plane type, fragments and solver effort vs time
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pczio  # noqa: E402

KIND = {0: "radial", 1: "transverse", 2: "circumferential", 3: "pellet interface"}


def _frame_list(series, spec):
    if not spec:
        n = len(series)
        return sorted(set([min(n - 1, max(1, n // 3)), min(n - 1, max(1, 2 * n // 3)), n - 1]))
    out = []
    for tok in spec.split(","):
        i = int(tok)
        out.append(i if i >= 0 else len(series) + i)
    return out


def _hist_at(hist, t):
    i = int(np.argmin(np.abs(hist["time_s"].to_numpy() - t)))
    return hist.iloc[i]


def _setup3d(ax, R, Ztop, z0=0.0):
    ax.set_box_aspect((1, 1, (Ztop - z0) / (2 * R)))
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(z0, Ztop)
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()


def draw_surface(ax, surf, warp, vlim, cmap):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as mcolors
    P = surf["points"] + warp * surf["point_data"]["displacement_m"]
    quads = P[surf["cells"]]
    val = surf["cell_data"]["max_principal_MPa"]
    norm = mcolors.Normalize(vmin=vlim[0], vmax=vlim[1])
    col = cmap(norm(val))
    pc = Poly3DCollection(quads, facecolors=col, edgecolors=(0, 0, 0, 0.15), linewidths=0.2)
    ax.add_collection3d(pc)
    return norm


def draw_cracks(ax, cr, warp, dcut, kinds=(0, 1, 2), cmap=None, omax=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as mcolors
    d = cr["cell_data"]["damage"]
    k = cr["cell_data"]["kind"]
    sel = (d >= dcut) & np.isin(k, kinds)
    P = cr["points"] + warp * cr["point_data"]["displacement_m"]
    quads = P[cr["cells"][sel]]
    op = np.clip(cr["cell_data"]["opening_um"][sel], 0, None)
    norm = mcolors.Normalize(vmin=0, vmax=omax or max(1e-3, np.percentile(op, 98) if len(op) else 1))
    if len(quads):
        pc = Poly3DCollection(quads, facecolors=cmap(norm(op)), edgecolors=(0, 0, 0, 0.25), linewidths=0.15)
        ax.add_collection3d(pc)
    return norm, int(sel.sum())


def outline(ax, R, Ztop, n_pellets, H):
    th = np.linspace(0, 2 * np.pi, 120)
    for z in [p * H for p in range(n_pellets + 1)]:
        ax.plot(R * np.cos(th), R * np.sin(th), z, color="0.45", lw=0.6)
    for a in (np.deg2rad(-58 + 90), np.deg2rad(-58 - 90)):
        ax.plot([R * np.cos(a)] * 2, [R * np.sin(a)] * 2, [0, Ztop], color="0.45", lw=0.6)


def fig_3d(run, frames, warp, dcut, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    man = run["manifest"]
    cfg = man.get("config", {})
    R = R0 = float(cfg.get("geometry.radius", 4.67e-3))
    H = float(cfg.get("geometry.height", 14e-3))
    npel = int(cfg.get("geometry.n_pellets", 1))
    Ztop = H * npel
    hist = run["history"]
    ss = pczio.series(run["dir"], "surface")
    cs = pczio.series(run["dir"], "cracks")
    idx = _frame_list(ss, frames)
    smaps = [pczio.read_vtu(ss[i][1]) for i in idx]
    cmaps = [pczio.read_vtu(cs[i][1]) for i in idx]
    # plot box large enough for the warped shapes
    wz = [(s["points"] + warp * s["point_data"]["displacement_m"])[:, 2] for s in smaps]
    wr = [np.hypot(*(s["points"] + warp * s["point_data"]["displacement_m"])[:, :2].T) for s in smaps]
    z0, Ztop, R = min(0.0, min(w.min() for w in wz)), max(Ztop, max(w.max() for w in wz)), max(R, max(w.max() for w in wr))
    vmax = max(np.percentile(s["cell_data"]["max_principal_MPa"], 99.5) for s in smaps)
    vmin = min(np.percentile(s["cell_data"]["max_principal_MPa"], 0.5) for s in smaps)
    vmax = max(vmax, 1.0)
    cm_s = plt.get_cmap("turbo")
    cm_c = plt.get_cmap("magma_r")
    omax = max([np.percentile(c["cell_data"]["opening_um"][c["cell_data"]["damage"] >= dcut], 98)
                if np.any(c["cell_data"]["damage"] >= dcut) else 0.01 for c in cmaps])
    n = len(idx)
    fig = plt.figure(figsize=(3.3 * n + 1.2, 7.6 * (0.55 + 0.45 * Ztop / (2 * R) / 1.5)))
    for j, (i, s, c) in enumerate(zip(idx, smaps, cmaps)):
        t = ss[i][0]
        h = _hist_at(hist, t)
        ax = fig.add_subplot(2, n, j + 1, projection="3d")
        _setup3d(ax, R, Ztop, z0)
        ns = draw_surface(ax, s, warp, (vmin, vmax), cm_s)
        ax.set_title(f"t = {t:.4g} s\nq' = {h.linear_power_Wpm / 100:.0f} W/cm,  ΔT = {h.dT_K:.0f} K", fontsize=9)
        ax2 = fig.add_subplot(2, n, n + j + 1, projection="3d")
        _setup3d(ax2, R, Ztop, z0)
        outline(ax2, R0, H * npel, npel, H)
        nc, cnt = draw_cracks(ax2, c, warp, dcut, cmap=cm_c, omax=omax)
        ax2.set_title(f"crack area {h.crack_area_mm2:.1f} mm²,  {int(h.n_fragments)} fragment(s)", fontsize=9)
    cax = fig.add_axes([0.93, 0.56, 0.012, 0.32])
    fig.colorbar(plt.cm.ScalarMappable(norm=ns, cmap=cm_s), cax=cax, label="max principal stress [MPa]")
    cax2 = fig.add_axes([0.93, 0.10, 0.012, 0.32])
    fig.colorbar(plt.cm.ScalarMappable(norm=nc, cmap=cm_c), cax=cax2, label="crack opening [µm]")
    fig.suptitle(f"pelletcz - deformed pellet (displacements x{warp:g}) and cohesive cracks (damage ≥ {dcut})",
                 fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.91, top=0.9, bottom=0.02, wspace=0.0, hspace=0.12)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig_midplane(run, frames, warp, dcut, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    cfg = run["manifest"].get("config", {})
    R = float(cfg.get("geometry.radius", 4.67e-3))
    H = float(cfg.get("geometry.height", 14e-3))
    bs = pczio.series(run["dir"], "bulk")
    cs = pczio.series(run["dir"], "cracks")
    idx = _frame_list(bs, frames)
    hist = run["history"]
    n = len(idx)
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n + 1, 3.6), squeeze=False)
    lim = 0
    data = []
    for i in idx:
        b = pczio.read_vtu(bs[i][1])
        c = pczio.read_vtu(cs[i][1])
        P = b["points"]
        U = b["point_data"]["displacement_m"]
        cells = b["cells"]
        cz = P[cells][:, :, 2].mean(axis=1)
        zl = np.unique(np.round(P[:, 2], 12))
        zb = zl[np.argmin(np.abs(zl - 0.5 * H))]
        lay = np.nonzero(np.abs(P[cells][:, 0, 2] - zb) < 1e-12)[0]
        if len(lay) == 0:
            lay = np.nonzero(np.abs(cz - 0.5 * H) == np.min(np.abs(cz - 0.5 * H)))[0]
        polys = (P + warp * U)[cells[lay][:, :4]][:, :, :2] * 1e3
        val = b["cell_data"]["hoop_stress_MPa"][lay]
        lim = max(lim, np.percentile(np.abs(val), 99))
        # crack traces in the slice plane
        cq = c["cells"]
        cp = c["points"] + warp * c["point_data"]["displacement_m"]
        zmin = c["points"][cq][:, :, 2].min(axis=1)
        sel = (c["cell_data"]["damage"] >= dcut) & np.isin(c["cell_data"]["kind"], [0, 2]) & (np.abs(zmin - zb) < 1e-12)
        segs = []
        for q in cq[sel]:
            pz = c["points"][q][:, 2]
            low = q[np.argsort(pz)[:2]]
            segs.append(cp[low][:, :2] * 1e3)
        data.append((bs[i][0], polys, val, segs))
    for ax, (t, polys, val, segs) in zip(axs[0], data):
        pc = PolyCollection(polys, array=val, cmap="RdBu_r", clim=(-lim, lim), edgecolors="none")
        ax.add_collection(pc)
        if segs:
            ax.add_collection(LineCollection(segs, colors="k", linewidths=1.6))
        ax.set_aspect("equal")
        ax.set_xlim(-1.05 * R * 1e3, 1.05 * R * 1e3)
        ax.set_ylim(-1.05 * R * 1e3, 1.05 * R * 1e3)
        h = _hist_at(hist, t)
        ax.set_title(f"t = {t:.4g} s, ΔT = {h.dT_K:.0f} K", fontsize=9)
        ax.set_xlabel("x [mm]")
    axs[0][0].set_ylabel("y [mm]")
    fig.colorbar(pc, ax=axs[0].tolist(), label="hoop stress [MPa]", shrink=0.85)
    fig.suptitle(f"mid-height cross-section (displacements x{warp:g}); black = cracked cohesive faces", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_history(run, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = run["history"]
    t = h.time_s.to_numpy()
    tt = np.where(t > 0, t, np.nan)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(tt, h.dT_K, color="C3", lw=1.8, label="ΔT centre - surface")
    ax.set_ylabel("ΔT [K]")
    ax2 = ax.twinx()
    ax2.plot(tt, h.linear_power_Wpm / 100, color="0.4", lw=1.2, ls="--", label="q'")
    ax2.set_ylabel("linear power [W/cm]")
    ax.set_xscale("log")
    ax.set_xlabel("time [s]")
    ax.set_title("thermal loading")
    s = pczio.read_json(os.path.join(run["dir"], "summary.json")) if os.path.exists(
        os.path.join(run["dir"], "summary.json")) else {}
    if s.get("onset_time_s"):
        ax.axvline(s["onset_time_s"], color="k", lw=0.8, ls=":")
        ax.text(s["onset_time_s"], ax.get_ylim()[1] * 0.95, f" first crack\n ΔT={s['onset_dT_K']:.0f} K", fontsize=8,
                va="top")
    ax = axs[0, 1]
    for col, lab in (("crack_area_radial_mm2", "radial"), ("crack_area_transverse_mm2", "transverse"),
                     ("crack_area_circumferential_mm2", "circumferential"), ("crack_area_mm2", "total")):
        ax.plot(h.dT_K, h[col], lw=2 if lab == "total" else 1.4, label=lab, color="k" if lab == "total" else None)
    ax.set_xlabel("ΔT [K]")
    ax.set_ylabel("crack area [mm²]")
    ax.set_title("crack surface area vs temperature difference")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 0]
    ax.plot(tt, h.crack_area_mm2, color="k", lw=1.8, label="crack area")
    ax.set_xscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("crack area [mm²]")
    axb = ax.twinx()
    axb.step(tt, h.n_fragments, color="C2", where="post", lw=1.2)
    axb.set_ylabel("fragments", color="C2")
    ax.set_title("crack growth in time (bursts at fixed load)")
    ax = axs[1, 1]
    ax.semilogy(np.arange(len(h)), np.diff(np.concatenate([[0], t])).clip(1e-9), color="C0", lw=1, label="dt [s]")
    ax.set_xlabel("step")
    ax.set_ylabel("time step [s]")
    axc = ax.twinx()
    axc.plot(np.arange(len(h)), h.newton_iters, color="C1", lw=0.8, alpha=0.8)
    axc.set_ylabel("Newton iterations", color="C1")
    ax.set_title("adaptive time step (µs during crack growth)")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--frames", default="", help="comma list of output indices (negative = from end)")
    ap.add_argument("--warp", type=float, default=150.0)
    ap.add_argument("--dcut", type=float, default=0.99)
    ap.add_argument("--out", default=None, help="output directory (default <run>/figures)")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    run = pczio.read_run(a.run_dir)
    od = a.out or os.path.join(a.run_dir, "figures")
    os.makedirs(od, exist_ok=True)
    fig_3d(run, a.frames, a.warp, a.dcut, os.path.join(od, a.prefix + "cracking_3d.png"))
    fig_midplane(run, a.frames, a.warp, a.dcut, os.path.join(od, a.prefix + "midplane.png"))
    fig_history(run, os.path.join(od, a.prefix + "history.png"))


if __name__ == "__main__":
    main()
