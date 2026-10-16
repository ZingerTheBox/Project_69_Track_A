import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize_scalar

from src import config as C
from src.pipeline import make_backend
from src.world import Disturbance, Scenario


def scenario(disturbances=(), wheel_err=(0.0, 0.0), track_err=0.0):
    return Scenario(seed=0, start=np.array([1.0, 2.5, 0.0]), prior=np.array([1.0, 2.5, 0.0]), obstacles=[],
                    surprise=None, disturbances=list(disturbances), wheel_err=wheel_err, track_err=track_err)


def drive(bk, v, w, secs):
    n = int(secs / C.DT_CTRL)
    p = [bk.truth()]
    enc = np.zeros(2)
    for _ in range(n):
        bk.step(v, w)
        enc += bk.read_encoders()
        p.append(bk.truth())
    p = np.array(p)
    speed = np.hypot(*np.diff(p[:, :2], axis=0).T) / C.DT_CTRL
    return np.arange(1, n + 1) * C.DT_CTRL, speed, enc, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="kinematic", choices=["kinematic", "mujoco"])
    a = ap.parse_args()
    bk = make_backend(a.backend)

    bk.reset(scenario())
    bk.read_encoders()
    t, sp, _, _ = drive(bk, 0.15, 0.0, 2.0)
    final = sp[-20:].mean()
    dt = C.DT_CTRL

    def avg_model(tau):
        k = np.arange(1, len(t) + 1)
        return final * (1 - tau / dt * (np.exp(-(k - 1) * dt / tau) - np.exp(-k * dt / tau)))

    tau = minimize_scalar(lambda x: np.sum((avg_model(x) - sp) ** 2), bounds=(0.005, 0.5), method="bounded").x
    print(f"[1] motor lag: steady speed {final:.3f} m/s, fitted first-order tau = {tau:.3f} s "
          f"(config.MOTOR_TAU = {C.MOTOR_TAU})")

    bk.reset(scenario())
    bk.read_encoders()
    _, _, enc, p = drive(bk, 0.15, 0.0, 6.5)
    d_true, d_enc = np.hypot(*(p[-1, :2] - p[0, :2])), enc.mean()
    print(f"[2a] straight: ground {d_true:.3f} m vs encoders {d_enc:.3f} m  -> scale error {100 * (d_enc / d_true - 1):+.2f} %")

    bk.reset(scenario())
    bk.read_encoders()
    _, _, enc, p = drive(bk, 0.0, 0.8, 2 * np.pi / 0.8 + 0.3)
    turn_true = np.unwrap(p[:, 2])[-1] - p[0, 2]
    turn_enc = (enc[1] - enc[0]) / C.TRACK
    print(f"[2b] spin: true {np.degrees(turn_true):.1f} deg vs odometry {np.degrees(turn_enc):.1f} deg "
          f"-> heading scale error {100 * (turn_enc / turn_true - 1):+.2f} %")

    bk.reset(scenario([Disturbance(0.0, 3.5, "slip", (0.35,))]))
    bk.read_encoders()
    _, _, enc, p = drive(bk, 0.18, 0.0, 3.0)
    d_true, d_enc = np.hypot(*(p[-1, :2] - p[0, :2])), enc.mean()
    print(f"[3] slip window: ground {d_true:.3f} m vs encoders {d_enc:.3f} m -> robot moved {100 * (1 - d_true / d_enc):.0f} % "
          f"less than the wheels say (target ~35 %)")


if __name__ == "__main__":
    main()
