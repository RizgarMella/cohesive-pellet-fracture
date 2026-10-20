#!/usr/bin/env python3
"""predict_fragments.py - crack area / fragment estimate from a power history, geometry and material,
using the equations found by fit_pysr.py (adapted from peridynamic-pellet-fracture).

  python tools/predict_fragments.py --history my_history.txt --R 4.67e-3 --H 14e-3 \
      --model studies/campaign-A/derived/pysr/model_log10_nfrag.json \
      --model studies/campaign-A/derived/pysr/model_log10_specific_area.json [--json out.json]

History file: rows "time_s q_W_per_cm" ('#' comments). Material defaults are the nominal UO2 values in
features.UO2; override any of them. Prints each estimate with a +-1 sigma band (validation residual sd)
and WARNS when an input lies outside the training domain (extrapolation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as ft  # noqa: E402


def evaluate(card: dict, feats: dict) -> float:
    import sympy as sp
    syms = {f: sp.Symbol(f) for f in card["features"]}
    expr = sp.sympify(card["expression"], locals=syms)
    fn = sp.lambdify([syms[f] for f in card["features"]], expr, "math")
    return float(fn(*[feats[f] for f in card["features"]]))


def read_history_file(path):
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            v = [float(x) for x in line.replace(",", " ").split()]
            rows.append((v[0], v[1]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--R", type=float, default=4.67e-3)
    ap.add_argument("--H", type=float, default=14e-3)
    for k, v in ft.UO2.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--json", help="also write results here")
    a = ap.parse_args()
    rows = read_history_file(a.history)
    geom = dict(R=a.R, H=a.H)
    mat = {k: getattr(a, k) for k in ft.UO2}
    f = ft.groups(geom, mat, rows)
    out = {"inputs": {"geometry": geom, "material": mat, "history": a.history}, "predictions": []}
    for path in a.model:
        card = json.load(open(path))
        y = evaluate(card, f)
        sd = card["residual_sd"]
        warn = [k for k in card["features"] if not (card["domain"][k][0] <= f[k] <= card["domain"][k][1])]
        name = card["target"]
        if card.get("target_transform") == "log10":
            name = name.replace("log10_", "")
            scale = a.R if name.endswith("_over_R") else 1.0
            val, lo, hi = (10 ** y) * scale, (10 ** (y - sd)) * scale, (10 ** (y + sd)) * scale
        else:
            val, lo, hi = y, y - sd, y + sd
        print(f"{name:>22s}: {val:.4g}  (+-1 sigma: {lo:.4g} - {hi:.4g})"
              + (f"   WARNING extrapolating in {warn}" if warn else ""))
        out["predictions"].append(dict(target=card["target"], value=val, lo_1sigma=lo, hi_1sigma=hi,
                                       extrapolated_features=warn, model=os.path.abspath(path),
                                       features={k: f[k] for k in card["features"]}))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
