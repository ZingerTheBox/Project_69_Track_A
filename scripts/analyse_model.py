import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C
from src.model import (controllability_matrix, discretise, error_model, error_model_with_lag,
                       lqr_gain, observability_rank, rank_and_sv)
from src.world import MARKER_BY_ID

np.set_printoptions(precision=4, suppress=True)


def check_controllability():
    print("=== Controllability of the tracking-error model ===")

    # a few (v_r, w_r) operating points, including standstill
    for v, w in [(0.0, 0.0), (0.03, 0.0), (0.18, 0.0), (0.0, 0.5), (0.18, 0.4)]:
        A, B = error_model(v, w)
        rank, sv = rank_and_sv(controllability_matrix(A, B))
        print(f"  v_r={v:4.2f} w_r={w:4.2f}  rank {rank}/3   sigma_min = {sv.min():.4f}")

    A, B = error_model_with_lag(0.18, 0.0)
    rank = rank_and_sv(controllability_matrix(A, B))[0]
    print(f"  with first-order motor lag (tau={C.MOTOR_TAU}s): rank {rank}/5")

    # closed loop poles for the nominal straight-line speed
    p = C.LQRParams()
    Q = np.diag([p.q_x, p.q_y, p.q_th])
    R = np.diag([p.r_v, p.r_w])
    K = lqr_gain(0.18, 0.0, Q, R)
    A, B = error_model(0.18, 0.0)
    Ad, Bd = discretise(A, B, C.DT_CTRL)
    poles = np.linalg.eigvals(Ad - Bd @ K)
    print(f"  LQR gain at v_r=0.18, w_r=0:\n{K}")
    print(f"  closed-loop poles (discrete): {np.round(poles, 3)}  |z|max={abs(poles).max():.3f}")


def check_observability():
    print("\n=== Observability of [x, y, theta] from camera range-bearing (linearised, windowed) ===")
    x0 = np.array([3.0, 2.5, 0.0])
    m0, m1, m2 = (MARKER_BY_ID[i].pos for i in (0, 4, 6))

    still = [(0.0, 0.0)] * 5
    arc = [(0.02, 0.025)] * 20

    cases = {
        "no marker (dead reckoning)": (still, [[] for _ in range(6)]),
        "1 marker, robot static": (still, [[m0]] * 6),
        "2 markers, robot static": (still, [[m0, m1]] * 6),
        "1 marker, robot driving an arc": (arc, [[m0]] * 21),
        "3 markers, robot driving": (arc, [[m0, m1, m2]] * 21),
    }

    results = {}
    for name, (steps, visible) in cases.items():
        rank, sv, _ = observability_rank(x0, steps, visible)
        if len(sv) < 3:
            sv = np.r_[sv, np.zeros(3 - len(sv))]  # pad so the plot always has 3 bars
        results[name] = sv
        print(f"  {name:34s} rank {rank}/3   singular values {np.round(sv, 4)}")

    print("\n  -> when the marker leaves the view the EKF reverts to the 'no marker' row: "
          "covariance grows with travel.")
    return results


def make_plot(results):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

    # left: how controllability falls off as the robot slows down
    speeds = np.linspace(0, 0.22, 45)
    sigma_min = [rank_and_sv(controllability_matrix(*error_model(v, 0.0)))[1].min() for v in speeds]
    ax[0].plot(speeds, sigma_min, lw=2, color="#1b9e77")
    ax[0].set(xlabel="reference speed $v_r$ [m/s]  ($w_r$=0)",
              ylabel=r"$\sigma_{min}$ of controllability matrix",
              title="Controllability degrades to zero at standstill")
    ax[0].grid(alpha=.3)

    # right: observability singular values for each case
    names = list(results)
    for k in range(3):
        vals = [max(results[n][k], 1e-9) for n in names]  # clip zeros so the log axis works
        ax[1].barh(np.arange(len(names)) + (k - 1) * 0.26, vals, 0.26, label=f"$\\sigma_{k + 1}$")
    ax[1].set_xscale("log")
    ax[1].set_yticks(range(len(names)), names, fontsize=8)
    ax[1].set(xlabel="singular value (log; 1e-9 = zero)",
              title="Observability: singular values of $\\mathcal{O}$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="x")

    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/model_analysis.png", dpi=160)
    print("\nwrote results/model_analysis.png")


def main():
    check_controllability()
    results = check_observability()
    make_plot(results)


if __name__ == "__main__":
    main()
