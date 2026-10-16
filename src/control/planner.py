"""
planner.py - global path planning and reference trajectory.

Occupancy grid -> inflate -> A* -> line-of-sight shortcut -> straight dock approach
-> Chaikin smoothing -> arc-length resample -> speed profile -> Trajectory.

Uses the estimated pose and a map of the static obstacles plus the LiDAR-detected
dynamic ones, never simulator ground truth.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .. import config as C

RES = 0.05
NX, NY = int(round(C.ROOM_W / RES)), int(round(C.ROOM_H / RES))
MARGIN = C.BODY_RADIUS + 0.10
_XC = (np.arange(NX) + 0.5) * RES
_YC = (np.arange(NY) + 0.5) * RES
_XX, _YY = np.meshgrid(_XC, _YC)


def cell_of(x, y):
    return (int(np.clip(y // RES, 0, NY - 1)), int(np.clip(x // RES, 0, NX - 1)))


class Grid:
    def __init__(self, circles, dynamic=None, margin=MARGIN):
        occ = np.zeros((NY, NX), bool)
        for cx, cy, r in circles:
            occ |= (_XX - cx) ** 2 + (_YY - cy) ** 2 <= r * r
        if dynamic is not None:
            occ |= dynamic
        padded = np.pad(occ, 1, constant_values=True)
        dist = distance_transform_edt(~padded)[1:-1, 1:-1] * RES
        self.dist = dist - RES / 2
        self.blocked = self.dist < margin

    def free(self, x, y):
        i, j = cell_of(x, y)
        return not self.blocked[i, j]

    def line_free(self, p, q):
        n = max(2, int(np.hypot(*(np.asarray(q) - p)) / (RES / 2)))
        for u in np.linspace(0, 1, n):
            x, y = (1 - u) * np.asarray(p) + u * np.asarray(q)
            if not self.free(x, y):
                return False
        return True


def astar(grid, start, goal):
    blocked = grid.blocked.copy()
    si, sj = cell_of(*start)
    gi, gj = cell_of(*goal)
    blocked &= ~((_XX - start[0]) ** 2 + (_YY - start[1]) ** 2 < 0.25**2)
    if blocked[gi, gj]:
        return None
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, 2**.5), (-1, 1, 2**.5), (1, -1, 2**.5), (1, 1, 2**.5)]

    def h(i, j):
        dx, dy = abs(i - gi), abs(j - gj)
        return (dx + dy) + (2**.5 - 2) * min(dx, dy)

    g = {(si, sj): 0.0}
    parent = {}
    pq = [(h(si, sj), 0.0, (si, sj))]
    closed = set()
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == (gi, gj):
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return [((j + 0.5) * RES, (i + 0.5) * RES) for i, j in reversed(path)]
        for di, dj, c in moves:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < NY and 0 <= nj < NX) or blocked[ni, nj]:
                continue
            ng = gc + c
            if ng < g.get((ni, nj), 1e18):
                g[(ni, nj)] = ng
                parent[(ni, nj)] = cur
                heapq.heappush(pq, (ng + h(ni, nj), ng, (ni, nj)))
    return None


def chaikin(pts, iters):
    for _ in range(iters):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        out = np.empty((2 * len(q), 2))
        out[0::2], out[1::2] = q, r
        pts = np.vstack([pts[:1], out, pts[-1:]])
    return pts


@dataclass
class Trajectory:
    t: np.ndarray
    xy: np.ndarray
    th: np.ndarray
    v: np.ndarray
    w: np.ndarray
    waypoints: np.ndarray

    @property
    def T(self):
        return float(self.t[-1])

    def sample(self, t):
        if t >= self.T:
            return (*self.xy[-1], self.th[-1], 0.0, 0.0)
        return (np.interp(t, self.t, self.xy[:, 0]), np.interp(t, self.t, self.xy[:, 1]),
                np.interp(t, self.t, self.th), np.interp(t, self.t, self.v), np.interp(t, self.t, self.w))

    def ahead(self, t, dist):
        i0 = int(np.searchsorted(self.t, t))
        if i0 >= len(self.t) - 1:
            return self.xy[-1:]
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(self.xy[i0:], axis=0).T))]
        return self.xy[i0:][s <= dist]


class Planner:
    def __init__(self, lqr=None):
        self.p = lqr or C.LQRParams()

    def plan(self, start_xy, goal_pose, circles, dynamic=None):
        grid = Grid(circles, dynamic)
        gx, gy, gth = goal_pose
        u = np.array([np.cos(gth), np.sin(gth)])
        pre = np.array([gx, gy]) - 0.9 * u
        cells = astar(grid, tuple(start_xy), tuple(pre))
        if cells is None:
            return None

        pts = [np.asarray(start_xy, float)]
        way = [np.array(c) for c in cells[1:]]
        i = 0
        while i < len(way):
            j = len(way) - 1
            while j > i and not grid.line_free(pts[-1], way[j]):
                j -= 1
            pts.append(way[j])
            i = j + 1
        pts[-1] = pre
        pts += [pre + 0.45 * u, np.array([gx, gy])]
        way = np.array(pts)

        smooth = way
        for it in (3, 2, 1, 0):
            cand = chaikin(way, it)
            fine = np.vstack([np.linspace(a, b, 12)[:-1] for a, b in zip(cand[:-1], cand[1:])])
            if all(grid.free(x, y) for x, y in fine[::2]):
                smooth = cand
                break
        return self._time_parameterise(smooth, way)

    def _time_parameterise(self, pts, waypoints):
        p = self.p
        seg = np.hypot(*np.diff(pts, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        n = max(int(s[-1] / p.ds), 2)
        sg = np.linspace(0, s[-1], n)
        xy = np.stack([np.interp(sg, s, pts[:, 0]), np.interp(sg, s, pts[:, 1])], axis=1)
        d = np.gradient(xy, sg, axis=0)
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        k = np.gradient(th, sg)
        k = np.convolve(k, np.ones(9) / 9, mode="same")

        v = np.minimum(p.v_cruise, np.sqrt(p.a_lat / np.maximum(np.abs(k), 1e-6)))
        v = np.minimum(v, 0.04 + 0.20 * (sg[-1] - sg))
        v = np.maximum(v, 0.03)
        for i in range(1, n):
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * p.a_long * p.ds))
        for i in range(n - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * p.a_long * p.ds))
        v[0] = min(v[0], 0.05)
        dt = p.ds / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)
        t = np.r_[0, np.cumsum(dt)]
        return Trajectory(t=t, xy=xy, th=th, v=v, w=k * v, waypoints=waypoints)
