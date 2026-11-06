from dataclasses import dataclass, field
import numpy as np
from . import config as C
from .control.planner import plan_path
from .utils import wrap

W, H = C.ROOM_W, C.ROOM_H
MARKER_Z = 0.15
MARKER_INSET = 0.005

@dataclass(frozen=True) #records
class Marker:
    id: int
    x: float
    y: float
    normal: float
    size: float

    @property
    def pos(self):
        return np.array([self.x, self.y])
    def corners_world(self):
        n = np.array([np.cos(self.normal), np.sin(self.normal), 0.0])
        right = np.array([-n[1], n[0], 0.0])
        up = np.array([0.0, 0.0, 1.0])
        c = np.array([self.x, self.y, MARKER_Z])
        h = self.size / 2
        return np.array([c - h * right + h * up, c + h * right + h * up,
                         c + h * right - h * up, c - h * right - h * up])

@dataclass
class Disturbance:
    t0: float
    t1: float
    kind: str #slip, push
    params: tuple = ()

@dataclass
class Scenario:
    seed: int
    start: np.ndarray
    prior: np.ndarray
    obstacles: list
    surprise: tuple
    disturbances: list = field(default_factory=list)
    marker_dropout: float = 0.0
    wheel_err: tuple = (0.0, 0.0)
    track_err: float = 0.0
    @property
    def all_obstacles(self):
        return list(self.obstacles) + ([self.surprise] if self.surprise else [])
#markers and docks
MARKERS = [Marker(0, W - MARKER_INSET, 2.5, np.pi, 0.15),
           Marker(1, W - MARKER_INSET, 1.0, np.pi, 0.15),
           Marker(2, W - MARKER_INSET, 4.0, np.pi, 0.15),
           Marker(3, 1.5, H - MARKER_INSET, -np.pi / 2, 0.15),
           Marker(4, 4.5, H - MARKER_INSET, -np.pi / 2, 0.15),
           Marker(5, 1.5, MARKER_INSET, np.pi / 2, 0.15),
           Marker(6, 4.5, MARKER_INSET, np.pi / 2, 0.15),
           Marker(7, MARKER_INSET, 1.5, 0.0, 0.15), 
           Marker(8, MARKER_INSET, 3.5, 0.0, 0.15)]
MARKER_BY_ID = {m.id: m for m in MARKERS}

def dock_pose():
    m = MARKER_BY_ID[C.DOCK_ID]
    return np.array([m.x + C.DOCK_STANDOFF * np.cos(m.normal),
                     m.y + C.DOCK_STANDOFF * np.sin(m.normal),
                     wrap(m.normal + np.pi)])

def obstacle_ok(o, start, obs): #random scenario
    ok = np.hypot(o[0] - start[0], o[1] - start[1]) > 1.0
    ok = ok and not (o[0] > 3.6 and abs(o[1] - 2.5) < 0.75)
    return ok and all(np.hypot(o[0] - p[0], o[1] - p[1]) > 1.1 for p in obs)

def place_surprise(start, obs):
    path = plan_path(start[:2], dock_pose(), obs)
    if path is None:
        return None
    xy = path["xy"]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    for frac in (0.45, 0.35, 0.55, 0.3, 0.6):
        i = int(np.searchsorted(s, frac * s[-1]))
        cand = (float(xy[i, 0]), float(xy[i, 1]), 0.15)
        far = all(np.hypot(cand[0] - p[0], cand[1] - p[1]) > p[2] + 0.65 for p in obs)
        far = far and np.hypot(*(np.array(cand[:2]) - start[:2])) > 1.0 and cand[0] < 4.2
        if far and plan_path(start[:2], dock_pose(), obs + [cand]) is not None:
            return cand
    return None

def make_scenario(seed, disturb=False, surprise=True, marker_dropout=0.0):
    rng = np.random.default_rng(seed)
    start = np.array([rng.uniform(0.7, 1.6), rng.uniform(1.0, 4.0), rng.uniform(-np.pi, np.pi)])
    prior = start + rng.normal(0, [C.P0_XY, C.P0_XY, C.P0_TH])
    obs = []
    while len(obs) < 3:
        o = (rng.uniform(2.0, 4.3), rng.uniform(0.9, 4.1), rng.uniform(0.14, 0.24))
        if obstacle_ok(o, start, obs):
            obs.append(o)
    sc = Scenario(seed=seed, start=start, prior=prior, obstacles=obs, surprise=None, marker_dropout=marker_dropout, wheel_err=tuple(rng.normal(0, 0.005, 2)), track_err=float(rng.normal(0, 0.01)))
    if surprise:
        sc.surprise = place_surprise(start, obs)
    if disturb:
        t_a, t_b = rng.uniform(8, 20), rng.uniform(25, 40)
        sc.disturbances = [
            Disturbance(t_a, t_a + 2.5, "slip", (0.35,)),
            Disturbance(t_b, t_b + 0.1, "push", (rng.uniform(-0.04, 0.04),
                                                 rng.uniform(-0.04, 0.04),
                                                 rng.uniform(-0.15, 0.15)))]
    return sc

def raycast(origin, angles, circles, max_range): #geometry
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    ox, oy = origin
    t = np.full(len(angles), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for pos, lim, comp in ((ox, W, 0), (ox, 0.0, 0), (oy, H, 1), (oy, 0.0, 1)):
            tt = (lim - pos) / d[:, comp]
            t = np.where(tt > 1e-9, np.minimum(t, tt), t)
    for cx, cy, r in circles:
        oc = np.array([ox - cx, oy - cy])
        b = d @ oc
        disc = b * b - (oc @ oc - r * r)
        hit = disc > 0
        tt = np.where(hit, -b - np.sqrt(np.where(hit, disc, 0)), np.inf)
        t = np.where((tt > 1e-9) & (tt < t), tt, t)
    return np.where(t <= max_range, t, np.inf)

def segment_blocked(p, q, circles, pad=0.0):
    d = q - p
    L2 = d @ d
    for cx, cy, r in circles:
        u = np.clip(((cx - p[0]) * d[0] + (cy - p[1]) * d[1]) / max(L2, 1e-12), 0, 1)
        if np.hypot(*(p + u * d - np.array([cx, cy]))) < r + pad:
            return True
    return False
