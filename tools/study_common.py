#!/usr/bin/env python3
"""study_common.py - cases, job files, run catalogue and labelled derived tables shared by every study.

STUDY LAYOUT (adapted from peridynamic-pellet-fracture; nothing is overwritten by a different build -
the build id is in every manifest):

  studies/<study>/
    study.json            study definition (levels, factors, seeds, base input)
    DATASET_CARD.md       human description of the dataset
    cases.csv             design matrix (one row per case, units in column names)
    inputs/base.ini       base input deck; each case = base + key=value overrides
    cases.txt             one line per case for  `pelletcz --ensemble cases.txt`  (MPI or serial)
    jobs/jobs.tsv         case_id <TAB> shell command (one line per task)
    jobs/slurm_array.sh   the same tasks as a Slurm array
    runs/<case_id>/       raw solver outputs (manifest, history, summary, vtu, csv)
    derived/*.csv + .meta.json   tidy analysis tables with a sidecar (columns, units, sources, script)
    figures/*.png         figures made only from derived/
    catalog.csv           run index: status, checksums, key metrics

Case ids are <prefix>-<6-digit index>-<6-hex hash of the overrides>, so a case id identifies its
inputs and re-planning the same design is idempotent.

    python tools/study_common.py run studies/meshconv --parallel 2 --threads 4   # local, resumable
    cd studies/meshconv && mpirun -n 8 pelletcz --ensemble cases.txt --threads 4  # MPI ensemble
    sbatch studies/meshconv/jobs/slurm_array.sh                                   # Slurm array
    python tools/study_common.py catalog studies/meshconv
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pczio  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_SOLVER = os.path.join(REPO, "build", "pelletcz")


@dataclass
class Case:
    case_id: str
    overrides: dict                                  # "section.key" -> value
    factors: dict = field(default_factory=dict)      # labelled design variables (units in names)
    split: str = "-"                                 # train / val / - (campaign)


def fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, (list, tuple)):
        return ",".join(fmt(x) for x in v)
    return str(v)


def table(rows) -> str:
    """[(t, v), ...] -> 't,v; t,v' (history tables in input files)"""
    return "; ".join(f"{fmt(float(t))},{fmt(float(v))}" for t, v in rows)


def case_id(prefix: str, index: int, overrides: dict) -> str:
    h = hashlib.sha256(json.dumps(overrides, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{prefix}-{index:06d}-{h}"


def override_args(ov: dict) -> list[str]:
    out = []
    for k, v in ov.items():
        s = fmt(v).replace(" ", "")
        out.append(f"{k}={s}")
    return out


def plan_study(study_dir: str, name: str, cases: list[Case], study_def: dict, base_input: str, *,
               threads: int = 0, card: str = "", vtu: bool = False, max_run_h: float = 24.0) -> None:
    os.makedirs(study_dir, exist_ok=True)
    for d in ("inputs", "jobs", "runs", "derived", "figures"):
        os.makedirs(os.path.join(study_dir, d), exist_ok=True)
    shutil.copyfile(base_input, os.path.join(study_dir, "inputs", "base.ini"))
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sdef = {"study": name, "created_utc": stamp, "n_cases": len(cases), "base_input": "inputs/base.ini", **study_def}
    with open(os.path.join(study_dir, "study.json"), "w") as f:
        json.dump(sdef, f, indent=1, default=str)
    rows, lines, ens = [], [], []
    for c in cases:
        ov = dict(c.overrides)
        ov["output.directory"] = f"runs/{c.case_id}"
        ov.setdefault("output.vtu", vtu)
        labels = {"study": name, "case_id": c.case_id, "split": c.split, **c.factors}
        ov["output.labels"] = ";".join(f"{k}={fmt(v)}" for k, v in labels.items())
        args = override_args(ov)
        ens.append("inputs/base.ini " + " ".join(args))
        cmd = '"${PELLETCZ:-pelletcz}" inputs/base.ini ' + " ".join(shlex.quote(a) for a in args)
        if threads:
            cmd = f"OMP_NUM_THREADS={threads} " + cmd
        cmd = f"mkdir -p runs/{c.case_id} && {cmd} > runs/{c.case_id}/stdout.log 2> runs/{c.case_id}/stderr.log"
        lines.append(f"{c.case_id}\t{cmd}")
        rows.append({"case_id": c.case_id, "split": c.split, **c.factors,
                     "overrides": " ".join(override_args(c.overrides))})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(study_dir, "cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(study_dir, "cases.txt"), "w") as f:
        f.write(f"# {name}: {len(cases)} cases - run from this directory:\n"
                f"#   mpirun -n <ranks> pelletcz --ensemble cases.txt --threads <omp threads per rank>\n")
        f.write("\n".join(ens) + "\n")
    with open(os.path.join(study_dir, "jobs", "jobs.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    hh = int(max_run_h)
    with open(os.path.join(study_dir, "jobs", "slurm_array.sh"), "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --array=1-{len(lines)}
#SBATCH --cpus-per-task={threads or 16}
#SBATCH --time={hh:02d}:00:00
#SBATCH --output=jobs/slurm_%A_%a.out
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
line=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" jobs/jobs.tsv)
id=$(echo "$line" | cut -f1)
[ -f "runs/$id/summary.json" ] && grep -q '"complete": true' "runs/$id/summary.json" && exit 0
bash -c "$(echo "$line" | cut -f2-)"
""")
    with open(os.path.join(study_dir, "DATASET_CARD.md"), "w") as f:
        f.write(f"# Dataset: {name}\n\nCreated {stamp} by `{os.path.basename(sys.argv[0])}`.\n\n{card.strip()}\n\n"
                f"* cases: {len(cases)} (`cases.csv`; each case = `inputs/base.ini` + overrides)\n"
                f"* raw runs: `runs/<case_id>/` (manifest.json, history.csv, summary.json, ...)\n"
                f"* derived tables: `derived/*.csv` with `.meta.json` sidecars\n"
                f"* licence / citation: _fill in before sharing_\n")
    print(f"planned {len(cases)} cases in {study_dir}")
    print(f"  local : python tools/study_common.py run {study_dir} --parallel 2 --threads 4")
    print(f"  MPI   : cd {study_dir} && mpirun -n <R> {DEFAULT_SOLVER} --ensemble cases.txt --threads <T>")
    print(f"  Slurm : cd {study_dir} && PELLETCZ={DEFAULT_SOLVER} sbatch jobs/slurm_array.sh")


def run_local(study_dir: str, parallel: int, threads: int, solver: str, only: list[str] | None = None) -> None:
    lines = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(study_dir, "jobs", "jobs.tsv")) if l.strip()]
    todo = [(cid, cmd) for cid, cmd in lines
            if not pczio.run_complete(os.path.join(study_dir, "runs", cid)) and (not only or cid in only)]
    print(f"{len(lines) - len(todo)} complete, {len(todo)} to run ({parallel} at a time, {threads} threads each)")
    env = dict(os.environ, PELLETCZ=solver, OMP_NUM_THREADS=str(threads))
    if parallel * threads > (os.cpu_count() or 1):
        # oversubscribed: spinning OpenMP barriers would stall every run
        env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        print(f"warning: {parallel} x {threads} threads > {os.cpu_count()} cores - using OMP_WAIT_POLICY=PASSIVE")

    def go(item):
        cid, cmd = item
        r = subprocess.run(cmd, shell=True, cwd=study_dir, env=env)
        ok = pczio.run_complete(os.path.join(study_dir, "runs", cid))
        print(f"  {cid}: {'complete' if ok else f'INCOMPLETE (exit {r.returncode})'}", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        res = list(ex.map(go, todo))
    print(f"done: {sum(res)}/{len(todo)} completed")


def _sha(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def catalog(study_dir: str):
    import pandas as pd
    cases = pd.read_csv(os.path.join(study_dir, "cases.csv"))
    rows = []
    for cid in cases.case_id:
        d = os.path.join(study_dir, "runs", cid)
        s = pczio.read_json(os.path.join(d, "summary.json")) if os.path.exists(os.path.join(d, "summary.json")) else {}
        status = "complete" if s.get("complete") else ("failed" if s else ("running" if os.path.isdir(d) and
                                                                            os.path.exists(os.path.join(d, "history.csv"))
                                                                            else "pending"))
        rows.append(dict(case_id=cid, status=status, message=s.get("message", ""), steps=s.get("steps"),
                         wall_s=s.get("wall_s"), onset_dT_K=s.get("onset_dT_K"),
                         crack_area_final_mm2=sum(s.get("crack_area_final_mm2") or [0]) if s else None,
                         sha_history=_sha(os.path.join(d, "history.csv")),
                         sha_summary=_sha(os.path.join(d, "summary.json")), path=f"runs/{cid}"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(study_dir, "catalog.csv"), index=False)
    print(df.status.value_counts().to_string())
    return df


UNITS = {"_mm2": "mm^2", "_mm3": "mm^3", "_mm": "mm", "_um": "um", "_m": "m", "_K": "K", "_s": "s", "_MPa": "MPa",
         "_Pa": "Pa", "_N": "N", "_Wpm": "W/m", "_Jpm2": "J/m^2", "_mJ": "mJ", "_J": "J", "_MWd_kgU": "MWd/kgU"}


def save_table(df, path: str, description: str, sources: list | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    cols = []
    for c in df.columns:
        unit = next((u for s, u in UNITS.items() if str(c).endswith(s) or f"{s}_" in str(c)), "")
        cols.append({"name": str(c), "unit": unit, "dtype": str(df[c].dtype)})
    meta = {"description": description, "columns": cols, "rows": int(len(df)),
            "produced_by": os.path.basename(sys.argv[0]), "created_utc":
                _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sources": sources or []}
    with open(path[:-4] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("study_dir")
    r.add_argument("--parallel", type=int, default=1)
    r.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    r.add_argument("--solver", default=DEFAULT_SOLVER)
    r.add_argument("--only", nargs="*")
    c = sp.add_parser("catalog")
    c.add_argument("study_dir")
    a = ap.parse_args()
    if a.cmd == "run":
        run_local(a.study_dir, a.parallel, a.threads, os.path.abspath(a.solver), a.only)
    else:
        catalog(a.study_dir)


if __name__ == "__main__":
    main()
