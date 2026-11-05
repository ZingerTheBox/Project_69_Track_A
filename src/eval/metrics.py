import numpy as np

from .. import config as C
from ..utils import wrap
from ..world import dock_pose
from .runlog import log_arrays

POS_TOL_CM = 5.0
HDG_TOL_DEG = 5.0


def nees_series(xh, P, gt):
    e = xh - gt
    e[:, 2] = wrap(e[:, 2])
    return np.einsum("ni,ni->n", e, np.linalg.solve(P + 1e-12 * np.eye(3), e[:, :, None])[:, :, 0])


def clearance_series(gt, circles):
    d = np.minimum.reduce([gt[:, 0], C.ROOM_W - gt[:, 0], gt[:, 1], C.ROOM_H - gt[:, 1]]) - C.BODY_RADIUS
    for cx, cy, r in circles:
        d = np.minimum(d, np.hypot(gt[:, 0] - cx, gt[:, 1] - cy) - r - C.BODY_RADIUS)
    return d


def summarise(log, sc):
    a = log_arrays(log)
    meta = log["meta"]
    m = dict(seed=sc.seed, n_replans=int(meta.get("n_replans", 0)), reached=bool(meta.get("reached", False)),
             backend=meta.get("backend", ""), had_obstacle_found=len(log["obstacles_found"]) > 0)
    if len(a["t"]) == 0:
        m.update(collided=False, dock_pos_err_cm=np.nan, dock_hdg_err_deg=np.nan, time_s=0.0,
                 est_rmse_pos_cm=np.nan, nees_mean=np.nan, nis_mean=np.nan, track_rmse_ey_cm=np.nan,
                 effort=np.nan, min_clearance_cm=np.nan, success=False)
        return m

    gt, xh, P, t = a["gt"], a["xh"].copy(), a["P"], a["t"]
    clr = clearance_series(gt, sc.all_obstacles)
    goal = dock_pose()
    pos_err = float(np.hypot(*(gt[-1, :2] - goal[:2])) * 100)
    hdg_err = float(abs(wrap(gt[-1, 2] - goal[2])) * 180 / np.pi)
    e = xh - gt
    up = np.array(log["updates"]) if log["updates"] else np.zeros((0, 6))
    nis_ok = up[up[:, 5] > 0, 4] if len(up) else np.zeros(0)
    tracking = a["phase"] == "TRACK"
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else C.DT_CTRL
    m.update(
        collided=bool((clr < 0).any()),
        dock_pos_err_cm=pos_err,
        dock_hdg_err_deg=hdg_err,
        time_s=float(t[-1]),
        est_rmse_pos_cm=float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2)) * 100),
        nees_mean=float(nees_series(xh, P, gt).mean()),
        nis_mean=float(nis_ok.mean()) if len(nis_ok) else np.nan,
        track_rmse_ey_cm=float(np.sqrt(np.mean(a["err"][tracking, 1] ** 2)) * 100) if tracking.any() else np.nan,
        effort=float(np.sum(a["cmd"][:, 0] ** 2 + a["cmd"][:, 1] ** 2) * dt),
        min_clearance_cm=float(clr.min() * 100),
        gated_frac=float(1 - (up[:, 5] > 0).mean()) if len(up) else np.nan,
    )
    m["success"] = bool(m["reached"] and not m["collided"] and pos_err <= POS_TOL_CM and hdg_err <= HDG_TOL_DEG)
    return m
