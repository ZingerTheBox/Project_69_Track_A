import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C
from src.perception.aruco import synth_measurements
from src.utils import wrap
from src.world import MARKER_BY_ID, make_scenario

np.set_printoptions(precision=4, suppress=True)


def collect(n_poses, use_mujoco, seed=0):
    rng = np.random.default_rng(seed)
    sc = make_scenario(seed, surprise=False)
    bk = None
    if use_mujoco:
        import mujoco
        from src.sim.mujoco_backend import MujocoBackend
        bk = MujocoBackend()
        bk.reset(sc)
    rows = []
    for _ in range(n_poses):
        pose = np.array([rng.uniform(0.5, 5.4), rng.uniform(0.5, 4.5), rng.uniform(-np.pi, np.pi)])
        if bk is None:
            meas = synth_measurements(pose, [], rng)
        else:
            bk.data.qpos[:2] = pose[:2]
            bk.data.qpos[3:7] = [np.cos(pose[2] / 2), 0, 0, np.sin(pose[2] / 2)]
            mujoco.mj_forward(bk.model, bk.data)
            meas = bk.camera()
        for mid, r, b in meas:
            d = MARKER_BY_ID[mid].pos - pose[:2]
            r_true, b_true = np.hypot(*d), wrap(np.arctan2(d[1], d[0]) - pose[2])
            rows.append((r_true, r - r_true, wrap(b - b_true)))
    return np.array(rows)


def fit_noise(d):
    edges = np.linspace(0.3, C.CAM_MAX_RANGE, 9)
    ctr, sr, sb, br = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d[:, 0] >= lo) & (d[:, 0] < hi)
        if m.sum() < 20:
            continue
        ctr.append(d[m, 0].mean())
        sr.append(d[m, 1].std())
        sb.append(d[m, 2].std())
        br.append(d[m, 1].mean())
    ctr, sr, sb, br = map(np.array, (ctr, sr, sb, br))
    a0, a2 = np.linalg.lstsq(np.c_[np.ones_like(ctr), ctr ** 2], sr, rcond=None)[0]
    b0, b1 = np.linalg.lstsq(np.c_[np.ones_like(ctr), ctr], sb, rcond=None)[0]
    return ctr, sr, sb, br, (a0, a2, b0, b1)


def report(fit, bias):
    a0, a2, b0, b1 = fit
    print(f"\n{'':22s}{'fitted':>10s}{'config':>10s}")
    print(f"{'range a0 [m]':22s}{a0:10.4f}{C.RNG_A0:10.4f}\n{'range a2 [m/m^2]':22s}{a2:10.4f}{C.RNG_A2:10.4f}")
    print(f"{'bearing b0 [rad]':22s}{b0:10.4f}{C.BRG_B0:10.4f}\n{'bearing b1 [rad/m]':22s}{b1:10.5f}{C.BRG_B1:10.5f}")
    print(f"mean range bias: {bias.mean() * 100:+.2f} cm  (config floor a0 covers it)")


def plot_fit(ctr, sr, sb, fit, path):
    a0, a2, b0, b1 = fit
    rr = np.linspace(0.3, C.CAM_MAX_RANGE, 50)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ctr, sr * 100, "o", label="measured std")
    ax[0].plot(rr, (a0 + a2 * rr ** 2) * 100, label="fit $a_0+a_2r^2$")
    ax[0].plot(rr, (C.RNG_A0 + C.RNG_A2 * rr ** 2) * 100, "--", label="config (used)")
    ax[0].set(xlabel="true range [m]", ylabel="range error std [cm]", title="Camera range noise")
    ax[1].plot(ctr, np.degrees(sb), "o", label="measured std")
    ax[1].plot(rr, np.degrees(b0 + b1 * rr), label="fit $b_0+b_1r$")
    ax[1].plot(rr, np.degrees(C.BRG_B0 + C.BRG_B1 * rr), "--", label="config (used)")
    ax[1].set(xlabel="true range [m]", ylabel="bearing error std [deg]", title="Camera bearing noise")
    for x in ax:
        x.grid(alpha=.3)
        x.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco", action="store_true")
    ap.add_argument("--poses", type=int, default=4000)
    a = ap.parse_args()

    d = collect(300 if a.mujoco else a.poses, a.mujoco)
    print(f"{len(d)} marker measurements, true range {d[:, 0].min():.2f}-{d[:, 0].max():.2f} m")
    ctr, sr, sb, br, fit = fit_noise(d)
    report(fit, br)
    name = "sensor_calibration_mujoco.png" if a.mujoco else "sensor_calibration_twin.png"
    plot_fit(ctr, sr, sb, fit, f"results/{name}")


if __name__ == "__main__":
    main()
