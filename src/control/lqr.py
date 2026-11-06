from functools import lru_cache
import numpy as np
from .. import config as C
from ..model import lqr_gain
from ..utils import wrap

V_BIN, W_BIN = 0.02, 0.10           # ms rads

def body_error(xh, ref):
    c, s = np.cos(xh[2]), np.sin(xh[2])
    dx, dy = ref[0] - xh[0], ref[1] - xh[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(ref[2] - xh[2])])


@lru_cache(maxsize=None)
def scheduled_gain(v_bin, w_bin, q_scale, r_scale):
    return lqr_gain(v_bin * V_BIN, w_bin * W_BIN, q_scale * C.Q_LQR, r_scale * C.R_LQR)


def lqr_track(xh, ref, q_scale=1.0, r_scale=1.0):
    e = body_error(xh, ref)
    Kr = scheduled_gain(int(round(ref[3] / V_BIN)), int(round(ref[4] / W_BIN)), q_scale, r_scale)
    du = -Kr @ e
    return float(ref[3] + du[0]), float(ref[4] + du[1]), e


def pid_track(xh, ref, integ):

    e = body_error(xh, ref)
    integ = np.clip(integ + e[:2] * C.DT_CTRL, -C.I_MAX, C.I_MAX)
    v = ref[3] * np.cos(e[2]) + C.KP_X * e[0] + C.KI_X * integ[0]
    w = ref[4] + ref[3] * (C.KP_Y * e[1] + C.KI_Y * integ[1]) + C.KP_TH * np.sin(e[2])
    return float(v), float(w), e, integ
