from __future__ import annotations

import numpy as np

from .. import config as C
from ..model import body_to_wheels, limit_command
from ..perception.aruco import synth_measurements
from ..world import Scenario, raycast

SUBSTEPS = 5
TRUE_ODOM_K = 6e-4  


class KinematicBackend:
    name = "kinematic"

    def reset(self, sc):
        self.sc = sc
        self.rng = np.random.default_rng(sc.seed + 1000)
        self.x = sc.start.astype(float).copy()
        self.t = 0.0
        self.w_act = np.zeros(2)  # actual wheel speds
        self.ang = np.zeros(2)  # wheel ang
        self.ticks_prev = np.zeros(2, dtype=np.int64)
        self.R_true = C.WHEEL_RADIUS * (1 + np.array(sc.wheel_err))
        self.L_true = C.TRACK * (1 + sc.track_err)
        self._pushed = [False] * len(sc.disturbances)

    def _slip_factor(self):
        for d in self.sc.disturbances:
            if d.kind == "slip" and d.t0 <= self.t < d.t1:
                s = d.params[0]
                return np.array([1 - s, 1 - 0.6 * s])
        return np.ones(2)

    def step(self, v, w):
        v, w = limit_command(v, w)
        cmd = np.array(body_to_wheels(v, w))
        dt = C.DT_CTRL / SUBSTEPS
        for _ in range(SUBSTEPS):
            self.w_act += (cmd - self.w_act) * (1 - np.exp(-dt / C.MOTOR_TAU))
            dang = self.w_act * dt
            self.ang += dang
            travel = dang * self.R_true * self._slip_factor()
            travel = travel + self.rng.normal(0, np.sqrt(TRUE_ODOM_K * np.abs(travel) + 1e-12))
            ds, dth = travel.sum() / 2, (travel[1] - travel[0]) / self.L_true
            thm = self.x[2] + dth / 2
            self.x += np.array([ds * np.cos(thm), ds * np.sin(thm), dth])
            self.t += dt
            for k, d in enumerate(self.sc.disturbances):  # pushes
                if d.kind == "push" and not self._pushed[k] and self.t >= d.t0:
                    self.x += np.array(d.params)
                    self._pushed[k] = True
        self.x[2] = C.wrap(self.x[2])

    def read_encoders(self):
        ticks = np.round(self.ang / (2 * np.pi) * C.ENCODER_TICKS).astype(np.int64)
        d = (ticks - self.ticks_prev) * 2 * np.pi / C.ENCODER_TICKS * C.WHEEL_RADIUS
        self.ticks_prev = ticks
        return float(d[0]), float(d[1])

    def camera(self):
        return synth_measurements(self.x, self.sc.all_obstacles, self.rng, self.sc.marker_dropout)

    def lidar(self):
        ang = self.x[2] + np.linspace(0, 2 * np.pi, C.LIDAR_BEAMS, endpoint=False)
        r = raycast(self.x[:2], ang, self.sc.all_obstacles, C.LIDAR_MAX)
        r = r + self.rng.normal(0, C.LIDAR_SIGMA0 + C.LIDAR_SIGMA1 * np.where(np.isfinite(r), r, 0))
        return np.where(np.isfinite(r) & (r >= C.LIDAR_MIN) & (r <= C.LIDAR_MAX), r, np.inf)

    def truth(self):
        return self.x.copy()

    def grab_frames(self):
        return None
