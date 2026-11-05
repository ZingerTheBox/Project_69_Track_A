#Standard examples:
#python main.py 
#python main.py --backend mujoco --video
#python main.py --backend mujoco --disturb --video --tag disturbed
#python main.py --estimator ukf --tracker pid

import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.metrics import summarise
from src.eval.plots import make_dashboard, plot_control, plot_estimation, plot_run_overview
from src.pipeline import run_trial
from src.world import make_scenario

SHOWN = ["success", "reached", "collided", "n_replans", "time_s", "dock_pos_err_cm", "dock_hdg_err_deg",
         "est_rmse_pos_cm", "nees_mean", "nis_mean", "track_rmse_ey_cm", "min_clearance_cm"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="kinematic", choices=["kinematic", "mujoco"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--disturb", action="store_true", help="wheel-slip window + pose push")
    ap.add_argument("--dropout", type=float, default=0.0, help="probability a visible marker is missed per frame")
    ap.add_argument("--estimator", default="ekf", choices=["ekf", "ukf", "odom"])
    ap.add_argument("--tracker", default="lqr", choices=["lqr", "pid"])
    ap.add_argument("--no-replan", action="store_true")
    ap.add_argument("--video", action="store_true", help="render the dashboard video (MP4 if ffmpeg exists, else GIF)")
    ap.add_argument("--gif", action="store_true", help="force GIF output")
    ap.add_argument("--stride", type=int, default=2, help="video: every n-th control tick (2 = 2x real time at 20 fps)")
    ap.add_argument("--viewer", action="store_true", help="open the live MuJoCo viewer (mujoco backend)")
    ap.add_argument("--tag", default=None, help="output sub-folder name")
    ap.add_argument("--out", default="results")
    return ap.parse_args()


def main():
    a = parse_args()
    tag = a.tag or f"{a.backend}_seed{a.seed}" + ("_disturbed" if a.disturb else "")
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    sc = make_scenario(a.seed, disturb=a.disturb, marker_dropout=a.dropout)
    print(f"[main] backend={a.backend} seed={a.seed} estimator={a.estimator} tracker={a.tracker} "
          f"disturb={a.disturb} dropout={a.dropout}  hidden obstacle: {sc.surprise is not None}")
    log = run_trial(sc, backend=a.backend, estimator=a.estimator, tracker=a.tracker, replan=not a.no_replan,
                    record_frames=a.video and a.backend == "mujoco", viewer=a.viewer,
                    progress=lambda t, ph: print(f"\r[main] t = {t:5.1f} s  {ph:<6}", end="", flush=True))
    print()

    m = summarise(log, sc)
    for k in SHOWN:
        print(f"  {k:18s} {m[k]:.3f}" if isinstance(m[k], float) else f"  {k:18s} {m[k]}")
    (out / "summary.json").write_text(json.dumps(m, indent=1, default=float))

    plot_run_overview(log, out / "overview.png", sc)
    plot_estimation(log, out / "estimation.png")
    plot_control(log, out / "control.png")
    print(f"[main] figures -> {out}/")
    if a.video:
        ext = "gif" if a.gif else "mp4"
        print(f"[main] rendering dashboard (stride {a.stride}) ...")
        make_dashboard(log, sc, out / f"dashboard.{ext}", fps=20, stride=a.stride)
        print(f"[main] video -> {out}/dashboard.{ext}")


if __name__ == "__main__":
    main()
