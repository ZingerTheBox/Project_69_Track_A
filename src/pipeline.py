import numpy as np
from . import config as C
from .control.lqr import body_error, lqr_track, pid_track
from .control.planner import inflate, is_free, plan_path, traj_ahead, traj_sample
from .estimation.ekf import P0, predict, update
from .eval.runlog import log_event, log_step, new_log
from .model import limit_command
from .perception.lidar import lidar_update, new_obstacle_map, obstacle_mask
from .utils import wrap
from .world import dock_pose

ALIGN_TOL = np.radians(8)
LOOKAHEAD = 1.5
REPLAN_COOLDOWN = 1.0 
MAX_REPLANS = 8
HOLD_TIME = 1.5 

def make_backend(name):
    if name == "mujoco":
        from .sim.mujoco_backend import MujocoBackend
        return MujocoBackend()
    from .sim.kinematic_backend import KinematicBackend
    return KinematicBackend()

def blocked_ahead(omap, traj, t_traj, planned):
    blocked = inflate(omap["known"], obstacle_mask(omap))
    fresh = [c for c in omap["circles"] if not any(np.hypot(c[0] - q[0], c[1] - q[1]) < 0.15 for q in planned)]
    ahead = traj_ahead(traj, t_traj, LOOKAHEAD)
    return bool(fresh) and not all(is_free(blocked, x, y) for x, y in ahead)

def run_trial(sc, backend="kinematic", estimator="ekf", tracker="lqr", replan=True,
              q_scale=C.Q_SCALE, r_scale=C.R_SCALE, lqr_q_scale=1.0, lqr_r_scale=1.0,
              record_frames=False, max_time=90.0, viewer=False, progress=None):
    bk = make_backend(backend) if isinstance(backend, str) else backend
    if viewer and hasattr(bk, "enable_viewer"):
        bk.enable_viewer()
    bk.reset(sc)

    xh, P = np.asarray(sc.prior, float).copy(), P0.copy()
    omap = new_obstacle_map(sc.obstacles) 
    goal = dock_pose()
    log = new_log(dict(seed=sc.seed, backend=bk.name, estimator=estimator, tracker=tracker, frames=[]))
    traj = plan_path(xh[:2], goal, omap["known"])
    if traj is None:
        log["meta"]["reached"] = False
        return log
    log["paths"].append((0.0, traj["waypoints"]))

    phase, t_traj, last_replan = "ALIGN", 0.0, -9.0
    planned = []
    integ = np.zeros(2) #PI
    n_replans, hold_t, dt = 0, 0.0, C.DT_CTRL

    for k in range(int(max_time / dt)):
        t = k * dt
        xh, P = predict(xh, P, bk.read_encoders(), estimator, q_scale)
        if k % C.CAMERA_EVERY == 0:
            for mid, r, b in bk.camera():
                xh, P, nis, ok = update(xh, P, mid, r, b, estimator, r_scale)
                log["updates"].append([t, mid, r, b, nis, float(ok)])
        if k % C.LIDAR_EVERY == 0:
            scan = bk.lidar()
            log["scans"].append((t, scan, xh.copy()))
            if lidar_update(omap, scan, xh, P):
                for c in omap["circles"]:
                    if not any(abs(c[0] - o[1]) < 1e-9 for o in log["obstacles_found"]):
                        log["obstacles_found"].append((t, *c))
                log_event(log, t, "obstacle detected")
            if (replan and phase == "TRACK" and omap["circles"] and t - last_replan > REPLAN_COOLDOWN
                    and n_replans < MAX_REPLANS and blocked_ahead(omap, traj, t_traj, planned)):
                new_traj = plan_path(xh[:2], goal, omap["known"], obstacle_mask(omap))
                last_replan = t
                if new_traj is None:
                    log_event(log, t, "replan failed")
                else:
                    traj, t_traj, phase = new_traj, 0.0, "ALIGN"
                    n_replans += 1
                    planned = list(omap["circles"])
                    log["paths"].append((t, traj["waypoints"]))
                    log_event(log, t, f"replan {n_replans}")
                  
        ref = traj_sample(traj, t_traj) #control
        e = body_error(xh, ref)
        if phase == "ALIGN":
            e_th = wrap(ref[2] - xh[2])
            v, w = 0.0, float(np.clip(2.0 * e_th, -0.8, 0.8))
            if abs(e_th) < ALIGN_TOL:
                phase = "TRACK"
        elif phase == "TRACK":
            if tracker == "lqr":
                v, w, e = lqr_track(xh, ref, lqr_q_scale, lqr_r_scale)
            else:
                v, w, e, integ = pid_track(xh, ref, integ)
            lag = e[0] - 0.05
            t_traj += dt * float(np.clip(1.0 - max(lag, 0.0) / 0.25, 0.15, 1.0))
            if t_traj >= traj["t"][-1]:
                phase = "HOLD"
        else:
            v, w = 0.0, 0.0
            hold_t += dt
        v, w = limit_command(v, w)

        log_step(log, t, bk.truth(), xh, P, ref, [v, w], phase, e)
        if record_frames and k % 10 == 0:
            fr = bk.grab_frames()
            if fr is not None:
                log["meta"]["frames"].append(dict(t=t, **fr))
        if progress and k % 100 == 0:
            progress(t, phase)

        if phase == "HOLD" and hold_t >= HOLD_TIME:
            log["meta"]["reached"] = True
            log_event(log, t, "docked")
            break
        bk.step(v, w)
    log["meta"]["n_replans"] = n_replans
    return log
