"""pczio.py - readers for pelletcz run directories (no VTK dependency).

    run = read_run("runs/single_pellet_demo")    # manifest, summary, history (pandas)
    g = read_vtu("runs/.../cracks_0003.vtu")     # dict: points, cells, cell_types, point_data, cell_data
    series("runs/...", "surface")                # [(time_s, path), ...] from the .pvd collection
"""
from __future__ import annotations

import base64
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

_DT = {"Float64": np.float64, "Float32": np.float32, "Int32": np.int32, "Int64": np.int64,
       "UInt8": np.uint8, "UInt32": np.uint32, "UInt64": np.uint64}


def _array(el) -> np.ndarray:
    dt = _DT[el.get("type")]
    ncomp = int(el.get("NumberOfComponents", "1"))
    txt = (el.text or "").strip()
    if el.get("format") == "binary":
        raw = base64.b64decode(txt)
        nb = int(np.frombuffer(raw[:8], dtype=np.uint64)[0])
        a = np.frombuffer(raw[8:8 + nb], dtype=dt).copy()
    else:
        a = np.array(txt.split(), dtype=dt) if txt else np.zeros(0, dt)
    return a.reshape(-1, ncomp) if ncomp > 1 else a


def read_vtu(path: str) -> dict:
    root = ET.parse(path).getroot()
    piece = root.find("UnstructuredGrid/Piece")
    pts = _array(piece.find("Points/DataArray"))
    cells = {a.get("Name"): _array(a) for a in piece.find("Cells")}
    off = cells["offsets"]
    conn = cells["connectivity"]
    npc = np.diff(np.concatenate([[0], off]))
    out = {"points": pts, "cell_types": cells["types"], "point_data": {}, "cell_data": {}}
    if len(npc) and np.all(npc == npc[0]):
        out["cells"] = conn.reshape(-1, npc[0])
    else:
        out["cells"] = np.split(conn, off[:-1])
    for sec, key in (("PointData", "point_data"), ("CellData", "cell_data")):
        s = piece.find(sec)
        if s is not None:
            for a in s:
                out[key][a.get("Name")] = _array(a)
    return out


def series(run_dir: str, kind: str = "surface") -> list[tuple[float, str]]:
    p = os.path.join(run_dir, f"{kind}.pvd")
    if not os.path.exists(p):
        files = sorted(f for f in os.listdir(run_dir) if f.startswith(kind + "_") and f.endswith(".vtu"))
        return [(float("nan"), os.path.join(run_dir, f)) for f in files]
    root = ET.parse(p).getroot()
    return [(float(d.get("timestep")), os.path.join(run_dir, d.get("file"))) for d in root.iter("DataSet")]


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def read_history(run_dir: str):
    import pandas as pd
    return pd.read_csv(os.path.join(run_dir, "history.csv"))


def run_complete(run_dir: str) -> bool:
    p = os.path.join(run_dir, "summary.json")
    if not os.path.exists(p):
        return False
    try:
        return bool(read_json(p).get("complete"))
    except Exception:
        return False


def read_run(run_dir: str) -> dict:
    out = {"dir": run_dir}
    for name in ("manifest", "summary"):
        p = os.path.join(run_dir, f"{name}.json")
        out[name] = read_json(p) if os.path.exists(p) else {}
    hp = os.path.join(run_dir, "history.csv")
    out["history"] = read_history(run_dir) if os.path.exists(hp) else None
    return out


def value_at(history, xcol: str, ycol: str, xs) -> list[float]:
    """y at the LAST step whose running maximum of x is <= each value in xs.

    Crack growth happens in bursts at (almost) constant load, so this returns
    the state after all growth that occurred up to load level v. NaN if the
    run never reached v."""
    x = history[xcol].to_numpy(float)
    y = history[ycol].to_numpy(float)
    xm = np.maximum.accumulate(x)
    out = []
    for v in np.atleast_1d(xs):
        if len(xm) == 0 or xm[-1] < v:
            out.append(float("nan"))
            continue
        idx = np.nonzero(xm <= v)[0]
        out.append(float(y[idx[-1]]) if len(idx) else float(y[0]))
    return out


def read_angular(path: str):
    import pandas as pd
    return pd.read_csv(path)
