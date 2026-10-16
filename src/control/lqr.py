"""
lqr.py - trajectory tracking controllers working on the EKF estimate.

LQR: gain-scheduled on (v_r, w_r). K comes from the discrete Riccati equation
of the linearised error model in model.py, cached per velocity bin.
PID: simple P/I law on the same body-frame error, for comparison.

Both take the estimated pose, not the true one.
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..model import lqr_gain


def body_error(x, ref):
    c, s = np.cos(x[2]), np.sin(x[2])
    dx, dy = ref[0] - x[0], ref[1] - x[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, C.wrap(ref[2] - x[2])])


class LQRTracker:
    name = "LQR"
    V_BIN, W_BIN = 0.02, 0.10

    def __init__(self, q_scale=1.0, r_scale=1.0, params=None):
        p = params or C.LQRParams()
        self.Q = q_scale * np.diag([p.q_x, p.q_y, p.q_th])
        self.R = r_scale * np.diag([p.r_v, p.r_w])
        self._cache = {}

    def gain(self, v_r, w_r):
        key = (round(v_r / self.V_BIN), round(w_r / self.W_BIN))
        if key not in self._cache:
            self._cache[key] = lqr_gain(key[0] * self.V_BIN, key[1] * self.W_BIN, self.Q, self.R)
        return self._cache[key]

    def command(self, x, ref):
        e = body_error(x, ref)
        K = self.gain(ref[3], ref[4])
        du = -K @ e
        return float(ref[3] + du[0]), float(ref[4] + du[1]), e


class PIDTracker:
    name = "PID"

    def __init__(self, kx=1.6, ky=6.0, kth=3.0, ki_x=0.15, ki_y=0.6, i_max=0.15, dt=C.DT_CTRL):
        self.k = (kx, ky, kth, ki_x, ki_y)
        self.i_max, self.dt = i_max, dt
        self.ix = self.iy = 0.0

    def command(self, x, ref):
        e = body_error(x, ref)
        kx, ky, kth, ki_x, ki_y = self.k

        self.ix = float(np.clip(self.ix + e[0] * self.dt, -self.i_max, self.i_max))
        self.iy = float(np.clip(self.iy + e[1] * self.dt, -self.i_max, self.i_max))

        v = ref[3] * np.cos(e[2]) + kx * e[0] + ki_x * self.ix
        w = ref[4] + ref[3] * (ky * e[1] + ki_y * self.iy) + kth * np.sin(e[2])
        return float(v), float(w), e
