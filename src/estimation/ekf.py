import numpy as np
from .. import config as C
from ..model import F_uni, G_uni, H_rb, f_uni, h_rb
from ..utils import wrap
from ..world import MARKER_BY_ID

P0 = np.diag([C.P0_XY**2, C.P0_XY**2, C.P0_TH**2]) #prior covariance

def odom_cov(xh, u, q_scale=C.Q_SCALE): # noise models
    var = np.array([C.ODOM_K * abs(u[0]) + C.ODOM_FLOOR**2,
                    C.ODOM_K * abs(u[1]) + C.ODOM_FLOOR**2])
    G = G_uni(xh, u)
    return q_scale * G @ np.diag(var) @ G.T

def meas_cov(r, r_scale=C.R_SCALE):
    return r_scale * np.diag([(C.RNG_A0 + C.RNG_A2 * r * r) ** 2,
                              (C.BRG_B0 + C.BRG_B1 * r) ** 2])

def rb_residual(y, y_hat):
    return np.array([y[0] - y_hat[0], wrap(y[1] - y_hat[1])])

def nis_of(nu, S):
    return float(nu @ np.linalg.solve(S, nu))
    
def ekf_predict(xh, P, u, f, F_jac, Qd): # EKF
    F = F_jac(xh, u)
    return f(xh, u), F @ P @ F.T + Qd

def ekf_update(xh_pred, P_pred, y, h, H_jac, Rn, residual=None, gate=None):
    y_hat = h(xh_pred)
    H = H_jac(xh_pred)
    nu = (y - y_hat) if residual is None else residual(y, y_hat)
    S = H @ P_pred @ H.T + Rn
    if gate is not None and nis_of(nu, S) > gate:
        return xh_pred, P_pred, nu, S, False
    K = P_pred @ H.T @ np.linalg.inv(S)
    xh = xh_pred + K @ nu
    I_KH = np.eye(len(xh_pred)) - K @ H
    P = I_KH @ P_pred @ I_KH.T + K @ Rn @ K.T
    return xh, 0.5 * (P + P.T), nu, S, True

N_UKF = 3 
ALPHA, BETA, KAPPA = 0.5, 2.0, 0.0
LAM = ALPHA**2 * (N_UKF + KAPPA) - N_UKF
WM = np.full(2 * N_UKF + 1, 0.5 / (N_UKF + LAM))
WC = WM.copy()
WM[0] = LAM / (N_UKF + LAM)
WC[0] = LAM / (N_UKF + LAM) + 1 - ALPHA**2 + BETA

def sigma_points(xh, P):
    L = np.linalg.cholesky((N_UKF + LAM) * P + 1e-12 * np.eye(N_UKF))
    return np.array([xh] + [xh + L[:, i] for i in range(N_UKF)] + [xh - L[:, i] for i in range(N_UKF)])

def mean_pose(X, W):
    m = W @ X
    m[2] = np.arctan2(W @ np.sin(X[:, 2]), W @ np.cos(X[:, 2]))
    return m

def ukf_predict(xh, P, u, f, Qd):
    X = np.array([f(s, u) for s in sigma_points(xh, P)])
    xm = mean_pose(X, WM)
    D = X - xm
    D[:, 2] = wrap(D[:, 2])
    return xm, (WC[:, None] * D).T @ D + Qd

def ukf_update(xh, P, y, h, Rn, gate=None):
    X = sigma_points(xh, P)
    Z = np.array([h(s) for s in X])
    zm = np.array([WM @ Z[:, 0], np.arctan2(WM @ np.sin(Z[:, 1]), WM @ np.cos(Z[:, 1]))])
    dZ, dX = Z - zm, X - xh
    dZ[:, 1], dX[:, 2] = wrap(dZ[:, 1]), wrap(dX[:, 2])
    S = (WC[:, None] * dZ).T @ dZ + Rn
    Pxz = (WC[:, None] * dX).T @ dZ
    nu = rb_residual(y, zm)
    if gate is not None and nis_of(nu, S) > gate:
        return xh, P, nu, S, False
    K = Pxz @ np.linalg.inv(S)
    P = P - K @ S @ K.T
    return xh + K @ nu, 0.5 * (P + P.T), nu, S, True

def predict(xh, P, u, kind="ekf", q_scale=C.Q_SCALE):
    Qd = odom_cov(xh, u, q_scale)
    if kind == "ukf":
        xh, P = ukf_predict(xh, P, u, f_uni, Qd)
    else:
        xh, P = ekf_predict(xh, P, u, f_uni, F_uni, Qd)
    xh[2] = wrap(xh[2])
    return xh, P

def update(xh, P, mid, r, b, kind="ekf", r_scale=C.R_SCALE):
    if kind == "odom":
        return xh, P, float("nan"), False
    mk = MARKER_BY_ID[mid].pos
    y, Rn = np.array([r, b]), meas_cov(r, r_scale)
    if kind == "ukf":
        xh, P, nu, S, ok = ukf_update(xh, P, y, lambda s: h_rb(s, mk), Rn, C.GATE_CHI2)
    else:
        xh, P, nu, S, ok = ekf_update(xh, P, y, lambda s: h_rb(s, mk), lambda s: H_rb(s, mk), Rn, rb_residual, C.GATE_CHI2)
    if ok:
        xh[2] = wrap(xh[2])
    return xh, P, nis_of(nu, S), ok
