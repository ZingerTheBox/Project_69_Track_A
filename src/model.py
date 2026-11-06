import numpy as np
from scipy.linalg import expm, solve_discrete_are
from . import config as C
from .utils import wrap

def limit_command(v, w): # kinematics
    w = float(np.clip(w, -C.W_MAX, C.W_MAX))
    v_cap = max(C.V_MAX - 0.5 * C.TRACK * abs(w), 0.0)
    v = float(np.clip(v, -v_cap, v_cap))
    return v, w

def body_to_wheels(v, w):
    return (v - 0.5 * C.TRACK * w) / C.WHEEL_RADIUS, (v + 0.5 * C.TRACK * w) / C.WHEEL_RADIUS

def twist_to_travel(v, w, dt=C.DT_CTRL):
    return (v - 0.5 * C.TRACK * w) * dt, (v + 0.5 * C.TRACK * w) * dt

def f_uni(xh, u): #ekf model
    dsl, dsr = u
    ds, dth = 0.5 * (dsl + dsr), (dsr - dsl) / C.TRACK
    thm = xh[2] + 0.5 * dth
    return np.array([xh[0] + ds * np.cos(thm),
                     xh[1] + ds * np.sin(thm),
                     xh[2] + dth])

def F_uni(xh, u):
    dsl, dsr = u
    ds, dth = 0.5 * (dsl + dsr), (dsr - dsl) / C.TRACK
    thm = xh[2] + 0.5 * dth
    return np.array([[1.0, 0.0, -ds * np.sin(thm)], [0.0, 1.0,  ds * np.cos(thm)], [0.0, 0.0, 1.0]])

def G_uni(xh, u):
    dsl, dsr = u
    ds, dth = 0.5 * (dsl + dsr), (dsr - dsl) / C.TRACK
    thm = xh[2] + 0.5 * dth
    c, s, L = np.cos(thm), np.sin(thm), C.TRACK
    return np.array([[0.5 * c + ds * s / (2 * L), 0.5 * c - ds * s / (2 * L)], [0.5 * s - ds * c / (2 * L), 0.5 * s + ds * c / (2 * L)], [-1 / L, 1 / L]])

def h_rb(xh, mk):
    dx, dy = mk[0] - xh[0], mk[1] - xh[1]
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - xh[2])])


def H_rb(xh, mk):
    d = mk - xh[:2]
    q = max(d @ d, 1e-9)
    r = np.sqrt(q)
    return np.array([[-d[0] / r, -d[1] / r, 0.0],
                     [ d[1] / q, -d[0] / q, -1.0]])

def error_model(v_r, w_r): #rtack error, lqr
    A = np.array([[0.0, w_r, 0.0],
                  [-w_r, 0.0, v_r],
                  [0.0, 0.0, 0.0]])
    B = np.array([[-1.0, 0.0],
                  [0.0, 0.0],
                  [0.0, -1.0]])
    return A, B

def error_model_with_lag(v_r, w_r, tau=C.MOTOR_TAU):
    A3, B3 = error_model(v_r, w_r)
    A = np.zeros((5, 5))
    A[:3, :3] = A3
    A[:3, 3:] = B3
    A[3:, 3:] = -np.eye(2) / tau
    B = np.zeros((5, 2))
    B[3:, :] = np.eye(2) / tau
    return A, B

def discretise(A, B, dt):
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]

def lqr_gain(v_r, w_r, Q, R, dt=C.DT_CTRL):
    v_r = max(abs(v_r), 0.03) * np.sign(v_r if v_r else 1.0)
    A, B = error_model(v_r, w_r)
    Ad, Bd = discretise(A, B, dt)
    P = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

def controllability_matrix(A, B): #contrl, oberv
    n = A.shape[0]
    blocks, M = [], B
    for _ in range(n):
        blocks.append(M)
        M = A @ M
    return np.hstack(blocks)

def rank_and_sv(M, rel_tol=1e-9):
    if M.size == 0:
        return 0, np.zeros(0)
    sv = np.linalg.svd(M, compute_uv=False)
    if sv.max() <= 1e-12:
        return 0, sv
    return int((sv > rel_tol * sv.max()).sum()), sv

def observability_rank(x0, steps, visible):
    xh, Phi, rows = np.asarray(x0, float).copy(), np.eye(3), []
    for k in range(len(visible)):
        for mk in visible[k]:
            rows.append(H_rb(xh, np.asarray(mk, float)) @ Phi)
        if k < len(steps):
            u = twist_to_travel(*steps[k])
            Phi = F_uni(xh, u) @ Phi
            xh = f_uni(xh, u)
    if not rows:
        return 0, np.zeros(0), np.zeros((0, 3))
    O = np.vstack(rows)
    rk, sv = rank_and_sv(O)
    return rk, sv, O
