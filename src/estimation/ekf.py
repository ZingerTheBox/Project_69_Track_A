from __future__ import annotations

import numpy as np

from .. import config as C
from ..world import MARKER_BY_ID


def _odom_noise(dsl, dsr, p):
    # wheel noise grows with distance moved, plus a small constant floor
    var_l = p.odom_k * abs(dsl) + p.odom_floor ** 2
    var_r = p.odom_k * abs(dsr) + p.odom_floor ** 2
    return np.array([var_l, var_r])


def _motion(x, dsl, dsr):
    # differential drive, using the midpoint heading for the position update
    ds = 0.5 * (dsl + dsr)
    dth = (dsr - dsl) / C.TRACK
    th_mid = x[2] + 0.5 * dth
    return np.array([
        x[0] + ds * np.cos(th_mid),
        x[1] + ds * np.sin(th_mid),
        x[2] + dth,
    ])


def _motion_jacobians(x, dsl, dsr):
    ds = 0.5 * (dsl + dsr)
    dth = (dsr - dsl) / C.TRACK
    th_mid = x[2] + 0.5 * dth
    c, s, L = np.cos(th_mid), np.sin(th_mid), C.TRACK

    # F: d(motion)/d(state), G: d(motion)/d(wheel distances)
    F = np.array([
        [1, 0, -ds * s],
        [0, 1, ds * c],
        [0, 0, 1.0],
    ])
    G = np.array([
        [0.5 * c + ds * s / (2 * L), 0.5 * c - ds * s / (2 * L)],
        [0.5 * s - ds * c / (2 * L), 0.5 * s + ds * c / (2 * L)],
        [-1 / L, 1 / L],
    ])
    return F, G


def _h(x, m):
    # expected range and bearing to marker m from pose x
    d = m - x[:2]
    rng = np.hypot(d[0], d[1])
    brg = C.wrap(np.arctan2(d[1], d[0]) - x[2])
    return np.array([rng, brg])


def _meas_cov(r, p):
    # camera noise gets worse the further away the marker is
    sig_r = p.rng_a0 + p.rng_a2 * r * r
    sig_b = p.brg_b0 + p.brg_b1 * r
    return p.r_scale * np.diag([sig_r ** 2, sig_b ** 2])


class DeadReckoning:
    name = "odom"

    def __init__(self, x0, params=None):
        self.p = params or C.EKFParams()
        self.x = np.asarray(x0, float).copy()
        self.P = np.diag([self.p.p0_xy ** 2, self.p.p0_xy ** 2, self.p.p0_th ** 2])

    def predict(self, dsl, dsr):
        F, G = _motion_jacobians(self.x, dsl, dsr)
        Q = self.p.q_scale * G @ np.diag(_odom_noise(dsl, dsr, self.p)) @ G.T
        self.P = F @ self.P @ F.T + Q
        self.x = _motion(self.x, dsl, dsr)
        self.x[2] = C.wrap(self.x[2])

    def update(self, mid, r, b):
        # no sensor correction, so this is plain dead reckoning
        return float("nan"), False


class EKF(DeadReckoning):
    name = "EKF"

    def update(self, mid, r, b):
        m = MARKER_BY_ID[mid].pos
        d = m - self.x[:2]
        q = max(d @ d, 1e-9)  # avoid dividing by zero if we're on top of the marker
        rho = np.sqrt(q)

        H = np.array([
            [-d[0] / rho, -d[1] / rho, 0.0],
            [d[1] / q, -d[0] / q, -1.0],
        ])
        R = _meas_cov(r, self.p)

        nu = np.array([r, b]) - _h(self.x, m)
        nu[1] = C.wrap(nu[1])
        S = H @ self.P @ H.T + R

        # chi-square gate on the normalised innovation, skip outliers
        nis = float(nu @ np.linalg.solve(S, nu))
        if nis > self.p.gate_chi2:
            return nis, False

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = C.wrap(self.x[2])

        # Joseph form keeps P symmetric and positive definite
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return nis, True


class UKF(DeadReckoning):
    name = "UKF"
    ALPHA, BETA, KAPPA = 0.5, 2.0, 0.0

    def __init__(self, x0, params=None):
        super().__init__(x0, params)
        n = 3
        lam = self.ALPHA ** 2 * (n + self.KAPPA) - n
        self.lam = lam

        # sigma point weights
        self.Wm = np.full(2 * n + 1, 0.5 / (n + lam))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + 1 - self.ALPHA ** 2 + self.BETA

    def _sigma(self, x, P):
        n = 3
        # tiny jitter so cholesky doesn't fail when P gets close to singular
        L = np.linalg.cholesky((n + self.lam) * P + 1e-12 * np.eye(n))
        pts = [x]
        for i in range(n):
            pts.append(x + L[:, i])
        for i in range(n):
            pts.append(x - L[:, i])
        return np.array(pts)

    @staticmethod
    def _mean_pose(S, W):
        # angles can't be averaged directly, so go through sin/cos
        m = W @ S
        m[2] = np.arctan2(W @ np.sin(S[:, 2]), W @ np.cos(S[:, 2]))
        return m

    def predict(self, dsl, dsr):
        _, G = _motion_jacobians(self.x, dsl, dsr)
        Q = self.p.q_scale * G @ np.diag(_odom_noise(dsl, dsr, self.p)) @ G.T

        X = np.array([_motion(s, dsl, dsr) for s in self._sigma(self.x, self.P)])
        xm = self._mean_pose(X, self.Wm)
        D = X - xm
        D[:, 2] = C.wrap(D[:, 2])

        self.P = (self.Wc[:, None] * D).T @ D + Q
        self.x = xm
        self.x[2] = C.wrap(self.x[2])

    def update(self, mid, r, b):
        m = MARKER_BY_ID[mid].pos
        X = self._sigma(self.x, self.P)
        Z = np.array([_h(s, m) for s in X])

        z_rng = self.Wm @ Z[:, 0]
        z_brg = np.arctan2(self.Wm @ np.sin(Z[:, 1]), self.Wm @ np.cos(Z[:, 1]))
        zm = np.array([z_rng, z_brg])

        dZ = Z - zm
        dX = X - self.x
        dZ[:, 1] = C.wrap(dZ[:, 1])
        dX[:, 2] = C.wrap(dX[:, 2])

        R = _meas_cov(r, self.p)
        S = (self.Wc[:, None] * dZ).T @ dZ + R
        Pxz = (self.Wc[:, None] * dX).T @ dZ

        nu = np.array([r, b]) - zm
        nu[1] = C.wrap(nu[1])

        nis = float(nu @ np.linalg.solve(S, nu))
        if nis > self.p.gate_chi2:
            return nis, False

        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = C.wrap(self.x[2])
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return nis, True
