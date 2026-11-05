#Obstacle detection from LiDAR by map differencing against the known walls and obstacles

# Unexplained returns are clustered circle-fitted and confirmed over CONFIRM_SCANS scans. The scan is placed with the estimated pose and the match tolerance grows with its uncertainty
# The obstacle map is a dict: known (x, y, r) list, circles (confirmed unknown), cand (unconfirmed).

import numpy as np

from .. import config as C
from ..control.planner import XX, YY

MIN_POINTS = 3
CLUSTER_GAP = 0.15               # m, max spacing within one cluster
CONFIRM_SCANS = 2
MATCH_DIST = 0.25                # m, candidate/circle association
R_MIN, R_MAX = 0.08, 0.35        # plausible obstacle radius [m]
R_NOMINAL = 0.15                 # radius assumed for a short arc [m]
MASK_PAD = 0.03                  # m, extra radius in the planner mask


def new_obstacle_map(known):
    return dict(known=[tuple(map(float, c)) for c in known], circles=[], cand=[])


def fit_circle(pts):
    #Algebraic (Kasa) circle fit. Returns (cx, cy, r) or None
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(axis=1)
    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    return (float(cx), float(cy), float(np.sqrt(r2))) if r2 > 0 else None


def unexplained(scan, xh, P, known):
    # World-frame returns not explained by the walls or known obstacles, with their beam indices
    ang = xh[2] + np.linspace(0, 2 * np.pi, C.LIDAR_BEAMS, endpoint=False)
    pts = np.c_[xh[0] + scan * np.cos(ang), xh[1] + scan * np.sin(ang)]
    sig = np.sqrt(max(np.linalg.eigvalsh(P[:2, :2]).max(), 0.0))
    tol = 0.06 + 3 * sig
    keep = np.isfinite(scan)
    x, y = pts[:, 0], pts[:, 1]
    with np.errstate(invalid="ignore"):
        keep &= (x > tol) & (x < C.ROOM_W - tol) & (y > tol) & (y < C.ROOM_H - tol) # not on a wall
        for cx, cy, r in known:
            keep &= np.abs(np.hypot(x - cx, y - cy) - r) > tol * 1.2 + 0.02 # not on a known cylinder
            keep &= np.hypot(x - cx, y - cy) > r                           # not inside one
    idx = np.flatnonzero(keep)
    return pts[idx], idx


def cluster_returns(pts, idx):
    #Group consecutive beams with small spacing, returning a list of index lists into pts.
    clusters, cur = [], [0]
    for i in range(1, len(pts)):
        if idx[i] - idx[i - 1] <= 3 and np.hypot(*(pts[i] - pts[i - 1])) < CLUSTER_GAP:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    wraps = idx[0] == 0 and idx[-1] == C.LIDAR_BEAMS - 1 and np.hypot(*(pts[0] - pts[-1])) < CLUSTER_GAP
    if len(clusters) > 1 and wraps:
        clusters[0] = clusters.pop() + clusters[0] # join across beam 0
    return clusters


def cluster_circle(cl_pts, xh):
    #Circle (cx, cy, r) for one cluster, or None. A short arc gets the nominal radius behind its centroid.
    fit = fit_circle(cl_pts)
    if fit is None:
        return None
    cx, cy, r = fit
    if R_MIN <= r <= R_MAX:
        return cx, cy, r
    m = cl_pts.mean(axis=0)
    direction = m - xh[:2]
    direction /= max(np.hypot(*direction), 1e-9)
    return m[0] + direction[0] * R_NOMINAL, m[1] + direction[1] * R_NOMINAL, R_NOMINAL


def track_circle(omap, circle):
    #Associate one detected circle with the map. Returns True if it is newly confirmed.
    cx, cy, r = circle
    if any(np.hypot(c[0] - cx, c[1] - cy) < MATCH_DIST for c in omap["circles"]):
        return False
    for cand in omap["cand"]:
        if np.hypot(cand[0] - cx, cand[1] - cy) < MATCH_DIST:
            n = cand[3]
            cand[:3] = [(cand[0] * n + cx) / (n + 1), (cand[1] * n + cy) / (n + 1), (cand[2] * n + r) / (n + 1)]
            cand[3] += 1
            if cand[3] >= CONFIRM_SCANS:
                omap["circles"].append(tuple(cand[:3]))
                omap["cand"].remove(cand)
                return True
            return False
    omap["cand"].append([cx, cy, r, 1])
    return False


def drop_stale(omap, seen):
    #Drop candidates not re-observed this scan.
    omap["cand"] = [c for c in omap["cand"] if any(np.hypot(c[0] - s[0], c[1] - s[1]) < MATCH_DIST for s in seen)]


def lidar_update(omap, scan, xh, P):
    #Process one scan with the estimated pose -> Returns True if a new obstacle was confirmed
    pts, idx = unexplained(scan, xh, P, omap["known"])
    if len(pts) < MIN_POINTS:
        drop_stale(omap, [])
        return False
    seen, new = [], False
    for cl in cluster_returns(pts, idx):
        if len(cl) < MIN_POINTS:
            continue
        circle = cluster_circle(pts[cl], xh)
        if circle is None:
            continue
        seen.append(circle)
        new = track_circle(omap, circle) or new
    drop_stale(omap, seen)
    return new


def obstacle_mask(omap):
    occ = np.zeros(XX.shape, bool)
    for cx, cy, r in omap["circles"]:
        occ |= (XX - cx) ** 2 + (YY - cy) ** 2 <= (r + MASK_PAD) ** 2
    return occ
