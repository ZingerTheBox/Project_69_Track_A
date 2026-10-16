"""
mujoco_backend.py - the MuJoCo physics backend (needs mujoco >= 3.1).

Same interface as KinematicBackend but everything comes from the engine. Physics
runs at 200 Hz and the controller at 20 Hz, so 10 physics steps per control tick.
Encoders are the wheel hinge angles quantised to ENCODER_TICKS per rev, the camera
is an offscreen render of "cam" -> cv2.aruco -> solvePnP, lidar is 360 mj_ray calls
from the "lidar" site (rays only see group 0, the world geoms).

Slip: wheel friction drops to SLIP_MU and a viscous drag on the chassis pushes the
wheels past their friction limit, so the encoders keep counting while the robot
moves ~35 % less. Push: instant shift / yaw kick on the free joint.

Headless: set MUJOCO_GL=egl (Linux GPU) or osmesa (software). On macOS the live
viewer needs `mjpython main.py --viewer`.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .. import config as C
from ..model import body_to_wheels, limit_command
from ..perception.aruco import ArucoDetector, pnp_range_bearing
from ..world import MARKER_BY_ID, Scenario, dock_pose
from . import mjcf

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("MuJoCo is not installed. Run:  pip install mujoco   (or use --backend kinematic)") from exc

SLIP_MU = 0.10
SLIP_DRAG = 7.0  # N s/m, gives ~35 % speed loss at 0.18 m/s with SLIP_MU. Re-tuned after the
                 # friction priority fix in mjcf.py, check with scripts/identify_platform.py


class MujocoBackend:
    name = "mujoco"
    _flip = None  # marker texture flip, probed once per process

    def __init__(self, realtime=False):
        self._tmp = tempfile.TemporaryDirectory(prefix="track_a_mjcf_")
        self.asset_dir = Path(self._tmp.name)
        self.detector = ArucoDetector()
        self.realtime = realtime
        self._want_viewer = False
        self._viewer = None
        self.renderer = None

    def enable_viewer(self):
        self._want_viewer = True

    def _load(self, sc, flip):
        xml = mjcf.build_xml(sc, self.asset_dir, flip)
        path = self.asset_dir / "scene.xml"
        path.write_text(xml)
        if self.renderer is not None:
            self.renderer.close()
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=C.IMG_H, width=C.IMG_W)
        m = self.model
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("wl", "wr")]
        self.wheel_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("wheel_l_geom", "wheel_r_geom")]
        self.mu0 = m.geom_friction[self.wheel_geoms, 0].copy()
        self.substeps = max(1, int(round(C.DT_CTRL / m.opt.timestep)))
        self.group_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # rays only see group 0

    def _probe_detects_dock(self):
        # teleport 1 m in front of the dock marker, render, see if cv2.aruco finds it
        d = self.data
        q0 = d.qpos.copy()
        mx = MARKER_BY_ID[C.DOCK_ID]
        d.qpos[:2] = [mx.x + np.cos(mx.normal) * 1.0, mx.y + np.sin(mx.normal) * 1.0]
        yaw = C.wrap(mx.normal + np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, d)
        self.renderer.update_scene(d, camera="cam")
        ok = C.DOCK_ID in self.detector.detect(self.renderer.render())
        d.qpos[:] = q0
        return ok

    def reset(self, sc):
        self.sc = sc
        self.rng = np.random.default_rng(sc.seed + 2000)
        flips = [MujocoBackend._flip] if MujocoBackend._flip else [C.MARKER_FLIP] + [f for f in mjcf.FLIPS if f != C.MARKER_FLIP]
        for f in flips:  # try flips until the marker is detected
            self._load(sc, f)
            if self._probe_detects_dock():
                if MujocoBackend._flip is None and f != C.MARKER_FLIP:
                    print(f"[mujoco] marker texture needs flip='{f}' - set MARKER_FLIP = '{f}' in src/config.py")
                MujocoBackend._flip = f
                break
        else:
            raise RuntimeError("cv2.aruco cannot detect the dock marker in any MuJoCo render. "
                               "Run  python scripts/check_setup.py  for a diagnosis.")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.ticks_prev = np.zeros(2, dtype=np.int64)
        self._pushed = [False] * len(sc.disturbances)
        self._slipping = False
        if self._want_viewer:
            import mujoco.viewer as mj_viewer  # aliased, a bare import here would make `mujoco` a local name
            self._viewer = mj_viewer.launch_passive(self.model, self.data)

    @property
    def t(self):
        return float(self.data.time)

    def _set_slip(self, on):
        if on != self._slipping:
            self.model.geom_friction[self.wheel_geoms, 0] = SLIP_MU if on else self.mu0
            self._slipping = on

    def _apply_pose_push(self, dx, dy, dyaw):
        d = self.data
        d.qpos[0] += dx
        d.qpos[1] += dy
        qz = np.array([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qz, d.qpos[3:7].copy())
        d.qpos[3:7] = out / np.linalg.norm(out)

    def step(self, v, w):
        v, w = limit_command(v, w)
        wl, wr = body_to_wheels(v, w)
        self.data.ctrl[:] = [wl, wr]
        t = self.t
        slip = any(d.kind == "slip" and d.t0 <= t < d.t1 for d in self.sc.disturbances)
        self._set_slip(slip)
        for k, d in enumerate(self.sc.disturbances):
            if d.kind == "push" and not self._pushed[k] and t >= d.t0:
                self._apply_pose_push(*d.params)
                self._pushed[k] = True
        t_wall = time.time()
        for _ in range(self.substeps):
            if slip:
                self.data.xfrc_applied[self.base_id, :2] = -SLIP_DRAG * self.data.qvel[:2]
            else:
                self.data.xfrc_applied[self.base_id, :2] = 0.0
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            if self.realtime:
                time.sleep(max(0.0, C.DT_CTRL - (time.time() - t_wall)))

    def read_encoders(self):
        ang = np.array([self.data.qpos[a] for a in self.qadr])
        ticks = np.round(ang / (2 * np.pi) * C.ENCODER_TICKS).astype(np.int64)
        d = (ticks - self.ticks_prev) * 2 * np.pi / C.ENCODER_TICKS * C.WHEEL_RADIUS
        self.ticks_prev = ticks
        return float(d[0]), float(d[1])

    def truth(self):
        # chassis pose [x, y, yaw] from the free joint, evaluation only
        q = self.data.qpos
        w, x, y, z = q[3:7]
        return np.array([q[0], q[1], np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])

    def _render(self, cam):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def camera(self):
        # render, detect, PnP. Same (id, range, bearing) output as the kinematic twin
        img = self._render("cam")
        out = []
        for mid, corners in self.detector.detect(img).items():
            if mid not in MARKER_BY_ID:
                continue
            if self.sc.marker_dropout and self.rng.random() < self.sc.marker_dropout:
                continue
            side = np.min(np.hypot(*(corners - np.roll(corners, 1, axis=0)).T))
            if side < C.CAM_MIN_SIDE_PX:
                continue
            rb = pnp_range_bearing(corners, MARKER_BY_ID[mid].size)
            if rb is not None and rb[0] <= C.CAM_MAX_RANGE:
                out.append((mid, *rb))
        return out

    def lidar(self):
        mujoco.mj_forward(self.model, self.data)
        origin = self.data.site_xpos[self.site_id].copy()
        yaw = self.truth()[2]
        geomid = np.zeros(1, dtype=np.int32)
        r = np.full(C.LIDAR_BEAMS, np.inf)
        for i, a in enumerate(yaw + np.linspace(0, 2 * np.pi, C.LIDAR_BEAMS, endpoint=False)):
            dist = mujoco.mj_ray(self.model, self.data, origin, np.array([np.cos(a), np.sin(a), 0.0]),
                                 self.group_mask, 1, -1, geomid)
            if dist >= 0:
                r[i] = dist + self.rng.normal(0, C.LIDAR_SIGMA0 + C.LIDAR_SIGMA1 * dist)
        return np.where((r >= C.LIDAR_MIN) & (r <= C.LIDAR_MAX), r, np.inf)

    def grab_frames(self):
        cam = self._render("cam")
        top = self._render("top")
        return dict(cam=cv2.resize(cam, (160, 120), interpolation=cv2.INTER_AREA), top=top)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
        if self.renderer is not None:
            self.renderer.close()
