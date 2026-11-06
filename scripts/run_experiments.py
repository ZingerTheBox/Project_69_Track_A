# repeated trials, ablations and failure diagnosis.
# Conditions: nominal, disturbed, dropout30/60, ukf, pid, odom_only, no_replan, plus a q_scale x r_scale sweep.
# Outputs in results/: experiments.csv, failures.csv, summary.csv/json, fig_*.png

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.eval.metrics import summarise
from src.eval.report_plots import make_all
from src.pipeline import run_trial
from src.world import make_scenario

CONDITIONS = {"nominal": {}, "disturbed": dict(disturb=True), "dropout30": dict(dropout=0.3),
              "dropout60": dict(dropout=0.6), "ukf": dict(estimator="ukf"), "pid": dict(tracker="pid"),
              "odom_only": dict(estimator="odom"), "no_replan": dict(replan=False)}
SWEEP_Q, SWEEP_R = (0.25, 1, 4, 16, 64), (0.25, 1, 4)


def run_one(job):
   #One trial to metrics dict. Top level so it can be pickled for multiprocessing
    cond, seed, kw, backend = job
    sc = make_scenario(seed, disturb=kw.get("disturb", False), marker_dropout=kw.get("dropout", 0.0))
    run_kw = {k: v for k, v in kw.items() if k in ("estimator", "tracker", "replan")}
    if "q" in kw:
        run_kw.update(q_scale=kw["q"], r_scale=kw["r"])
    m = summarise(run_trial(sc, backend=backend, **run_kw), sc)
    m.update(condition=cond, **{k: v for k, v in kw.items() if k in ("q", "r")})
    m["had_surprise"] = sc.surprise is not None
    return m


def diagnose(m):
    # Rule-based cause label for a failing trial
    if m["collided"] and m["n_replans"] == 0:
        return "collision: unknown obstacle never replanned (LiDAR detection missed or replan disabled)"
    if m["collided"]:
        return "collision after replan: path margin / localisation error near an obstacle"
    if not m["reached"]:
        return "timeout or no path"
    if m["dock_pos_err_cm"] > 5:
        return "final pose error: estimator bias or tracker lag at the dock"
    if m["dock_hdg_err_deg"] > 5:
        return "final heading error: heading estimate poorly constrained by a single marker"
    return "unclassified"


def make_jobs(n, backend, quick):
    jobs = [(c, s, kw, backend) for c, kw in CONDITIONS.items() for s in range(n)]
    if not quick:
        for q, r in itertools.product(SWEEP_Q, SWEEP_R):
            jobs += [(f"sweep_q{q}_r{r}", s, dict(q=q, r=r), backend) for s in range(min(n, 12))]
    return jobs


def summary_table(df):
    g = df[~df.condition.str.startswith("sweep")].groupby("condition")
    return g.agg(n=("seed", "count"), success=("success", "mean"), collisions=("collided", "mean"),
                 dock_pos_cm=("dock_pos_err_cm", "mean"), dock_hdg_deg=("dock_hdg_err_deg", "mean"),
                 time_s=("time_s", "mean"), est_rmse_cm=("est_rmse_pos_cm", "mean"), nees=("nees_mean", "mean"),
                 nis=("nis_mean", "mean"), track_ey_cm=("track_rmse_ey_cm", "mean"), effort=("effort", "mean"),
                 min_clear_cm=("min_clearance_cm", "min")).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--backend", default="kinematic", choices=["kinematic", "mujoco"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="skip the noise sweep")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    jobs = make_jobs(a.n, a.backend, a.quick)
    print(f"{len(jobs)} trials on '{a.backend}' with {a.workers} workers")
    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(run_one, jobs, chunksize=4))
    else:
        rows = [run_one(j) for j in jobs]
    print(f"done in {time.time() - t0:.0f} s")

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "experiments.csv", index=False)
    fails = df[~df.success].copy()
    fails["diagnosis"] = [diagnose(r) for r in fails.to_dict("records")]
    fails.to_csv(out / "failures.csv", index=False)

    summ = summary_table(df)
    print(summ.to_string())
    summ.to_csv(out / "summary.csv")
    (out / "summary.json").write_text(json.dumps(summ.reset_index().to_dict("records"), indent=1, default=float))
    make_all(df, out)
    print(f"figures + tables in {out}/")
    if len(fails):
        print(f"\n{len(fails)} failed trials - see failures.csv:")
        print(fails.groupby(["condition", "diagnosis"]).size().to_string())


if __name__ == "__main__":
    main()
