"""features.py - dimensionless inputs and targets for the crack/fragment metamodel.

ONE implementation used by campaign.py (dataset), fit_pysr.py (training) and predict_fragments.py
(end users), so a prediction is always computed from exactly the features the model was trained on.
(Adapted from peridynamic-pellet-fracture/features.py for the cohesive-zone model: the cohesive law is
defined by strength AND toughness, so the Irwin length l_ch = E G_c / sigma^2 is a material property here
rather than a calibration choice.)

Inputs: power history rows (time_s, q_W_per_cm), geometry (R, H, contact fraction), material (E, nu,
alpha, conductivity scale, mean strength, mean toughness, Weibull m, correlation length, rho).

Dimensionless groups
  Pi_T      = alpha E dT0 / ((1-nu) sigma)     peak elastic surface hoop stress (x2) / mean strength,
              dT0 = q'_max / (4 pi k_ref), k_ref = k(1000 K) of 95 % TD UO2 x conductivity scale
  Pi_R      = R / l_ch                          size / Irwin length (brittleness)
  Pi_H      = H / (2R)                          aspect ratio
  Pi_lc     = lc / R                            Weibull correlation length / size
  weibull_m, nu, rho                            already dimensionless
  Pi_rate   = log10(t_ramp / (R^2 / kappa))     ramp time / thermal diffusion time
  Pi_ramp2  = second power ramp / q'_max        (paper: +50 % over 30 min)
  Pi_bu     = end-of-history burnup / 50 MWd/kgU
Targets (log10)
  log10_specific_area (A R / V), log10_nfrag, log10_Leq_over_R (L_eq = 3V/A), Pi_T_onset (Pi_T at the
  first crack: how much of the strength is mobilised at initiation)
"""
from __future__ import annotations

import math

import numpy as np

FEATURES = ["Pi_T", "Pi_R", "Pi_H", "Pi_lc", "weibull_m", "nu", "rho", "Pi_rate", "Pi_ramp2", "Pi_bu"]
TARGETS = ["log10_specific_area", "log10_nfrag", "log10_Leq_over_R", "Pi_T_onset"]

UO2 = dict(E=219e9, nu=0.345, alpha=10.0e-6, k_scale=1.0, sigma=130e6, Gc=4.0, weibull_m=8.0, lc=0.5e-3, rho=0.5,
           density=10970 * 0.95, cp=300.0)


def k_ref(k_scale: float = 1.0) -> float:
    t = 1.0
    return (100.0 / (7.5408 + 17.692 * t + 3.6142 * t * t) + 6400.0 / t ** 2.5 * math.exp(-16.35 / t)) * k_scale


def burnup_MWd_kgU(rows, R: float, density: float = UO2["density"]) -> float:
    h = np.asarray(rows, float)
    t, q = h[:, 0], h[:, 1] * 100.0  # W/m
    energy = float(np.trapezoid(q, t)) if hasattr(np, "trapezoid") else float(np.trapz(q, t))  # J per m of rod
    mass_U = math.pi * R * R * density * 0.8815  # kgU per m
    return energy / mass_U / 8.64e10


def history_features(rows) -> dict:
    h = np.asarray(rows, float)
    t, q = h[:, 0], h[:, 1]
    qmax = float(q.max())
    if qmax <= 0:
        raise ValueError("history has no power")
    pos = np.nonzero(q > 0)[0]
    i0 = max(0, int(pos[0]) - 1)          # start of the first ramp
    i = i0
    while i < len(q) - 1 and q[i + 1] > q[i]:
        i += 1                              # end of the first ramp (first plateau)
    q1 = float(q[i])
    return dict(q_max_Wpcm=qmax, t_ramp_s=max(float(t[i] - t[i0]), 1e-3), ramp2_Wpcm=max(0.0, qmax - q1),
                duration_s=float(t[-1] - t[0]))


def groups(geom: dict, mat: dict, rows) -> dict:
    m = dict(UO2, **{k: v for k, v in mat.items() if v is not None})
    hf = history_features(rows)
    R, H = geom["R"], geom["H"]
    kr = k_ref(m["k_scale"])
    dT0 = hf["q_max_Wpcm"] * 100.0 / (4 * math.pi * kr)
    lch = m["E"] * m["Gc"] / m["sigma"] ** 2
    kappa = kr / (m["density"] * m["cp"])
    bu = burnup_MWd_kgU(rows, R, m["density"])
    return dict(hf, Pi_T=m["alpha"] * m["E"] * dT0 / ((1 - m["nu"]) * m["sigma"]), Pi_R=R / lch, Pi_H=H / (2 * R),
                Pi_lc=m["lc"] / R, weibull_m=m["weibull_m"], nu=m["nu"], rho=m["rho"],
                Pi_rate=math.log10(hf["t_ramp_s"] / (R * R / kappa)), Pi_ramp2=hf["ramp2_Wpcm"] / hf["q_max_Wpcm"],
                Pi_bu=bu / 50.0, dT0_K=dT0, l_ch_m=lch, burnup_MWd_kgU=bu)


def targets(summary: dict, geom: dict, mat: dict, body: int = 0) -> dict:
    m = dict(UO2, **{k: v for k, v in mat.items() if v is not None})
    fr = (summary.get("fragments_final") or [{}])[body]
    R, H = geom["R"], geom["H"]
    V = math.pi * R * R * H
    A = (summary.get("crack_area_final_mm2") or [float("nan")])[body] * 1e-6
    n = fr.get("n") or 0
    L = fr.get("Leq_mm")
    f = lambda v: math.log10(v) if v and v > 0 and math.isfinite(v) else float("nan")
    dTon = summary.get("onset_dT_K")
    pit_on = (m["alpha"] * m["E"] * dTon / ((1 - m["nu"]) * m["sigma"])) if dTon else float("nan")
    return dict(log10_specific_area=f(A * R / V), log10_nfrag=f(n), log10_Leq_over_R=f(L * 1e-3 / R) if L else float("nan"),
                Pi_T_onset=pit_on, nfrag=n, Leq_mm=L, crack_area_mm2=A * 1e6, onset_dT_K=dTon)
