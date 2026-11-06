import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt
from .. import config as C

RES = 0.05                                   # grid sol
NX, NY = int(round(C.ROOM_W / RES)), int(round(C.ROOM_H / RES))
MARGIN = C.BODY_RADIUS + 0.10                # rob radius + 10
XC = (np.arange(NX) + 0.5) * RES
YC = (np.arange(NY) + 0.5) * RES
XX, YY = np.meshgrid(XC, YC)                 # cell centres

def cell_of(x, y):
    return int(np.clip(y // RES, 0, NY - 1)), int(np.clip(x // RES, 0, NX - 1))


def inflate(circles, dynamic=None, margin=MARGIN):
    occ = np.zeros((NY, NX), bool)
    for cx, cy, r in circles:
        occ |= (XX - cx) ** 2 + (YY - cy) ** 2 <= r * r
    if dynamic is not None:
        occ |= dynamic
    padded = np.pad(occ, 1, constant_values=True) 
    dist = distance_transform_edt(~padded)[1:-1, 1:-1] * RES - RES / 2
    return dist < margin


def is_free(blocked, x, y):
    i, j = cell_of(x, y)
    return not blocked[i, j]


def line_free(blocked, p, q):
    n = max(2, int(np.hypot(*(np.asarray(q) - p)) / (RES / 2)))
    for u in np.linspace(0, 1, n):
        x, y = (1 - u) * np.asarray(p) + u * np.asarray(q)
        if not is_free(blocked, x, y):
            return False
    return True


def astar(blocked, start, goal):
    blocked = blocked.copy()
    si, sj = cell_of(*start)
    gi, gj = cell_of(*goal)
    blocked &= ~((XX - start[0]) ** 2 + (YY - start[1]) ** 2 < 0.25**2)
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


def shortcut(blocked, start_xy, cells):
    pts = [np.asarray(start_xy, float)]
    way = [np.array(c) for c in cells[1:]]
    i = 0
    while i < len(way):
        j = len(way) - 1
        while j > i and not line_free(blocked, pts[-1], way[j]):
            j -= 1
        pts.append(way[j])
        i = j + 1
    return pts


def chaikin(pts, iters):
    for _ in range(iters):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        out = np.empty((2 * len(q), 2))
        out[0::2], out[1::2] = q, r
        pts = np.vstack([pts[:1], out, pts[-1:]])
    return pts


def cut_corners(blocked, way):
    for it in (3, 2, 1, 0):
        cand = chaikin(way, it)
        fine = np.vstack([np.linspace(a, b, 12)[:-1] for a, b in zip(cand[:-1], cand[1:])])
        if all(is_free(blocked, x, y) for x, y in fine[::2]):
            return cand
    return way

def time_parameterise(pts, waypoints):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(s[-1] / C.DS), 2)
    sg = np.linspace(0, s[-1], n)
    xy = np.stack([np.interp(sg, s, pts[:, 0]), np.interp(sg, s, pts[:, 1])], axis=1)
    d = np.gradient(xy, sg, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = np.gradient(th, sg)
    k = np.convolve(k, np.ones(9) / 9, mode="same")

    v = np.minimum(C.V_CRUISE, np.sqrt(C.A_LAT / np.maximum(np.abs(k), 1e-6)))
    v = np.minimum(v, 0.04 + 0.20 * (sg[-1] - sg))
    v = np.maximum(v, 0.03)
    
    for i in range(1, n):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * C.A_LONG * C.DS))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * C.A_LONG * C.DS))
        
    v[0] = min(v[0], 0.05)
    dt = C.DS / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)
    t = np.r_[0, np.cumsum(dt)]
    return dict(t=t, xy=xy, th=th, v=v, w=k * v, waypoints=waypoints)

def traj_sample(traj, t):
    if t >= traj["t"][-1]:
        return (*traj["xy"][-1], traj["th"][-1], 0.0, 0.0)
    return (np.interp(t, traj["t"], traj["xy"][:, 0]), np.interp(t, traj["t"], traj["xy"][:, 1]),
            np.interp(t, traj["t"], traj["th"]), np.interp(t, traj["t"], traj["v"]),
            np.interp(t, traj["t"], traj["w"]))


def traj_ahead(traj, t, dist):
    i0 = int(np.searchsorted(traj["t"], t))
    if i0 >= len(traj["t"]) - 1:
        return traj["xy"][-1:]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(traj["xy"][i0:], axis=0).T))]
    return traj["xy"][i0:][s <= dist]

def plan_path(start_xy, goal_pose, circles, dynamic=None):
    blocked = inflate(circles, dynamic)
    gx, gy, gth = goal_pose
    u = np.array([np.cos(gth), np.sin(gth)])
    pre = np.array([gx, gy]) - 0.9 * u
    cells = astar(blocked, tuple(start_xy), tuple(pre))
    if cells is None:
        return None
    pts = shortcut(blocked, start_xy, cells)
    pts[-1] = pre
    pts += [pre + 0.45 * u, np.array([gx, gy])]
    way = np.array(pts)
    return time_parameterise(cut_corners(blocked, way), way)
