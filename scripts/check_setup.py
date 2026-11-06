#Checks packages, ffmpeg, a kinematic-twin trial, and the MuJoCo model, physics, camera, LiDAR and slip.

import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

results = []
mj = {} # MuJoCo backend and scenario shared between checks


def check(name, fn, hint=""):
    try:
        msg = fn()
        results.append((name, "PASS", msg or ""))
    except ImportError as e:
        results.append((name, "SKIP", f"{e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}   {hint}"))
    r = results[-1]
    print(f"[{r[1]}] {r[0]:38s} {r[2]}")


def pkgs():
    versions = {m: importlib.import_module(m).__version__ for m in ("numpy", "scipy", "matplotlib", "cv2")}
    import cv2
    assert hasattr(cv2, "aruco"), "cv2.aruco missing - pip install opencv-contrib-python (or opencv-python>=4.8)"
    return " ".join(f"{k} {x}" for k, x in versions.items())


def ffmpeg():
    p = shutil.which("ffmpeg")
    if not p:
        raise ImportError("ffmpeg not found - dashboard falls back to GIF (install ffmpeg for MP4)")
    return p


def twin():
    from src.eval.metrics import summarise
    from src.pipeline import run_trial
    from src.world import make_scenario
    sc = make_scenario(3)
    m = summarise(run_trial(sc), sc)
    assert m["success"], f"trial failed: {m}"
    return f"docked, pos err {m['dock_pos_err_cm']:.2f} cm"


def mj_backend():
    import mujoco
    from src.sim.mujoco_backend import MujocoBackend
    from src.world import make_scenario
    bk = MujocoBackend()
    sc = make_scenario(3, surprise=True)
    bk.reset(sc)
    mj.update(bk=bk, sc=sc)
    return f"mujoco {mujoco.__version__}, offscreen render OK, marker flip = '{MujocoBackend._flip}'"


def mj_physics():
    from src import config as C
    bk, sc = mj["bk"], mj["sc"]
    bk.reset(sc)
    x0 = bk.truth()
    for _ in range(20):
        bk.step(0.0, 0.0)
    z = bk.data.qpos[2]
    assert abs(z - C.WHEEL_RADIUS) < 0.006, f"chassis height {z:.4f} m (expected ~{C.WHEEL_RADIUS}) - tipping/sinking?"
    for _ in range(40):
        bk.step(0.10, 0.0)
    x1 = bk.truth()
    d = np.array([np.cos(x0[2]), np.sin(x0[2])])
    dxy = x1[:2] - x0[:2]
    along, lateral = dxy @ d, d[0] * dxy[1] - d[1] * dxy[0]
    assert 0.14 < along < 0.22, f"travelled {along:.3f} m, expected ~0.18 (wheel friction / actuator gain?)"
    assert abs(lateral) < 0.03, f"drifted {lateral:.3f} m sideways"
    return f"stands at z={z * 1000:.1f} mm, drove {along:.3f} m, lateral drift {lateral * 100:.1f} cm"


def mj_camera():
    import mujoco
    from src import config as C
    from src.world import MARKER_BY_ID
    bk = mj["bk"]
    m = MARKER_BY_ID[C.DOCK_ID]
    pose = np.array([m.x - 1.0, m.y, 0.0])
    bk.data.qpos[:2] = pose[:2]
    bk.data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(bk.model, bk.data)
    dets = {i: (r, b) for i, r, b in bk.camera()}
    assert C.DOCK_ID in dets, "dock marker not detected from 1 m in front of it"
    r, b = dets[C.DOCK_ID]
    assert abs(r - 1.0) < 0.03 and abs(b) < 0.03, f"PnP gave range {r:.3f} m, bearing {np.degrees(b):.2f} deg (truth 1.000 / 0)"
    return f"marker 0 detected, range {r:.3f} m (truth 1.000), bearing {np.degrees(b):+.2f} deg"


def mj_lidar():
    from src import config as C
    from src.world import raycast
    bk, sc = mj["bk"], mj["sc"]
    bk.reset(sc)
    pose = bk.truth()
    scan = bk.lidar()
    ang = pose[2] + np.linspace(0, 2 * np.pi, C.LIDAR_BEAMS, endpoint=False)
    ref = raycast(pose[:2], ang, sc.all_obstacles, C.LIDAR_MAX)
    both = np.isfinite(scan) & np.isfinite(ref)
    err = np.abs(scan[both] - ref[both])
    assert both.sum() > 50 and np.median(err) < 0.03 and np.percentile(err, 95) < 0.08, \
        f"{both.sum()} comparable beams, median err {np.median(err):.3f} m, 95 % {np.percentile(err, 95):.3f} m"
    return f"{both.sum()} beams agree with analytic ray-cast (median err {np.median(err) * 100:.1f} cm)"


def mj_slip():
    from src.world import Disturbance, make_scenario
    bk = mj["bk"]
    sc = make_scenario(3, surprise=False)
    sc.obstacles = []
    sc.start = np.array([1.0, 2.5, 0.0])
    sc.disturbances = [Disturbance(0.0, 3.0, "slip", (0.35,))]
    bk.reset(sc)
    bk.read_encoders()
    p0, enc = bk.truth(), np.zeros(2)
    for _ in range(40):
        bk.step(0.18, 0.0)
        enc += bk.read_encoders()
    ground = np.hypot(*(bk.truth()[:2] - p0[:2]))
    loss = 1 - ground / enc.mean()
    assert 0.15 < loss < 0.6, f"slip loss {loss * 100:.0f} % - tune SLIP_MU / SLIP_DRAG in mujoco_backend.py"
    return f"ground {ground:.3f} m vs encoders {enc.mean():.3f} m ({loss * 100:.0f} % loss)"


if __name__ == "__main__":
    print("Track A environment check\n")
    check("python packages", pkgs, "pip install -r requirements.txt")
    check("ffmpeg (optional, for MP4)", ffmpeg)
    check("kinematic twin: full docking trial", twin)
    check("MuJoCo model + offscreen rendering", mj_backend,
          "headless? set MUJOCO_GL=egl (GPU) or osmesa (software) before running")
    if mj:
        check("MuJoCo physics: stands + drives straight", mj_physics)
        check("MuJoCo camera: marker detection + PnP", mj_camera, "try MARKER_FLIP = 'ud' / 'lr' / 'both' in src/config.py")
        check("MuJoCo LiDAR vs analytic ray-cast", mj_lidar)
        check("MuJoCo disturbance: wheel slip", mj_slip)
    n_fail = sum(r[1] == "FAIL" for r in results)
    print(f"\n{'ALL GOOD' if not n_fail else f'{n_fail} check(s) FAILED'} - "
          f"{sum(r[1] == 'PASS' for r in results)} passed, {sum(r[1] == 'SKIP' for r in results)} skipped")
    sys.exit(1 if n_fail else 0)
