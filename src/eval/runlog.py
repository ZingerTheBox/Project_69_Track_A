import numpy as np


def new_log(meta=None):
    return dict(meta=meta or {}, paths=[], updates=[], scans=[], obstacles_found=[], events=[], rows=[])


def log_event(log, t, text):
    log["events"].append((float(t), text))


def log_step(log, t, gt, xh, P, ref, cmd, phase, err):
    log["rows"].append((float(t), np.array(gt, float), np.array(xh, float), np.array(P, float),
                        np.array(ref, float), np.array(cmd, float), phase, np.array(err, float)))


def log_arrays(log):
    if not log["rows"]:
        z = np.zeros((0, 3))
        return dict(t=np.zeros(0), gt=z, xh=z, P=np.zeros((0, 3, 3)), ref=np.zeros((0, 5)),
                    cmd=np.zeros((0, 2)), phase=np.array([], dtype=str), err=z)
    t, gt, xh, P, ref, cmd, phase, err = zip(*log["rows"])
    return dict(t=np.array(t), gt=np.array(gt), xh=np.array(xh), P=np.array(P), ref=np.array(ref),
                cmd=np.array(cmd), phase=np.array(phase), err=np.array(err))
