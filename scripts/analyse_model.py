import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src import config as C
from src.model import (controllability_matrix, discretise, error_model, error_model_with_lag, lqr_gain, observability_rank, rank_and_sv)
from src.world import MARKER_BY_ID

np.set_printoptions(precision=4, suppress=True)

def controllability():
    print("Controllability of the tracking-error model")
    for v, w in [(0.0, 0.0), (0.03, 0.0), (0.18, 0.0), (0.0, 0.5), (0.18, 0.4)]:
        A, B = error_model(v, w)
        rk, sv = rank_and_sv(controllability_matrix(A, B))
        print(f"  v_r={v:4.2f} w_r={w:4.2f}  rank {rk}/3   sigma_min = {sv.min():.4f}")
    A, B = error_model_with_lag(0.18, 0.0)
    print(f"  with first order motor lag (tau={C.MOTOR_TAU}s): rank {rank_and_sv(controllability_matrix(A, B))[0]}/5")

    Kr = lqr_gain(0.18, 0.0, C.Q_LQR, C.R_LQR)
    A, B = error_model(0.18, 0.0)
    Ad, Bd = discretise(A, B, C.DT_CTRL)
    cl = np.linalg.eigvals(Ad - Bd @ Kr)
    print(f"  LQR gain at v_r=0.18, w_r=0:\n{Kr}")
    print(f"  closed-loop poles (discrete): {np.round(cl, 3)}  |z|max={abs(cl).max():.3f}")

def observability():
    print("\n Observability of [x, y, theta] from camera range-bearing (linearised, windowed) ")
    x0 = np.array([3.0, 2.5, 0.0])
    m0, m1, m2 = (MARKER_BY_ID[i].pos for i in (0, 4, 6))
    still = [(0.0, 0.0)] * 5
    arc = [(0.02, 0.025)] * 20
    cases = {"no marker (dead reckoning)": (still, [[] for _ in range(6)]),
             "1 marker, robot static": (still, [[m0]] * 6),
             "2 markers, robot static": (still, [[m0, m1]] * 6),
             "1 marker, robot driving an arc": (arc, [[m0]] * 21),
             "3 markers, robot driving": (arc, [[m0, m1, m2]] * 21)}
    results = {}
    for name, (steps, visible) in cases.items():
        rk, sv, _ = observability_rank(x0, steps, visible)
        results[name] = np.r_[sv, np.zeros(3 - len(sv))]
        print(f"  {name:34s} rank {rk}/3   singular values {np.round(results[name], 4)}")
    print("\n  -> when the marker leaves the view the EKF reverts to the 'no marker' row:")
    print("     covariance grows with travel.")
    return results

def plot_analysis(results, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    vs = np.linspace(0, 0.22, 45)
    sigma_min = [rank_and_sv(controllability_matrix(*error_model(v, 0.0)))[1].min() for v in vs]
    ax[0].plot(vs, sigma_min, lw=2, color="#1b9e77")
    ax[0].set(xlabel="reference speed $v_r$ [m/s]  ($w_r$=0)", ylabel=r"$\sigma_{min}$ of controllability matrix",
              title="Controllability degrades to zero at standstill")
    names = list(results)
    for k in range(3):
        ax[1].barh(np.arange(len(names)) + (k - 1) * 0.26, [max(results[n][k], 1e-9) for n in names], 0.26,
                   label=f"$\\sigma_{k + 1}$")
    ax[1].set_xscale("log")
    ax[1].set_yticks(range(len(names)), names, fontsize=8)
    ax[1].set(xlabel="singular value (log; 1e-9 = zero)", title="Observability: singular values of $\\mathcal{O}$")
    ax[1].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].grid(alpha=.3, axis="x")
    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")

if __name__ == "__main__":
    controllability()
    plot_analysis(observability(), "results/model_analysis.png")
