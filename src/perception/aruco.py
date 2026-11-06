#camera perception. ArUco detection and solvePnP give (range, bearing) of a marker in the base frame.
#Corners come from cv2.aruco on MuJoCo renders, or from a synthetic pinhole projection with pixel noise (kinematic twin). Both go through the same PnP.

from functools import lru_cache

import cv2
import numpy as np

from .. import config as C
from ..world import MARKERS, segment_blocked

CELL_PX = 32            # texture pixels per ArUco cell
CORNER_BIAS_PX = 0.155  # detected quads are ~0.3 px small, which biases the PnP depth (scripts/calibrate_sensor.py)


# marker texture and detection
@lru_cache(maxsize=None)
def aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))


def marker_image(marker_id, flip="none"):
    #6x6 ArUco plus one white cell of margin.
    img = cv2.aruco.generateImageMarker(aruco_dictionary(), marker_id, 6 * CELL_PX, borderBits=1)
    img = np.pad(img, CELL_PX, constant_values=255)
    if flip in ("ud", "both"):
        img = img[::-1]
    if flip in ("lr", "both"):
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


@lru_cache(maxsize=None)
def aruco_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dictionary(), params)


def detect_markers(rgb):
    #Detect ArUco markers in an RGB image and returns {id: corners (4, 2) in pixels}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    corners, ids, _ = aruco_detector().detectMarkers(gray)
    if ids is None:
        return {}
    out = {}
    for i, c in zip(ids.ravel(), corners):
        q = c.reshape(4, 2).astype(float)
        u = q - q.mean(axis=0)
        out[int(i)] = q + CORNER_BIAS_PX * np.sqrt(2) * u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return out


def min_side_px(corners):
    return np.min(np.hypot(*(corners - np.roll(corners, 1, axis=0)).T))


# pixels -> (range, bearing)
def object_points(size):
    #Marker corners: TL, TR, BR, BL
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def pnp_range_bearing(corners, size):
    ok, rvec, tvec = cv2.solvePnP(object_points(size), np.asarray(corners, np.float64).reshape(4, 1, 2),
                                  C.K_CAM, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    xc, _, zc = tvec.ravel()                    # marker centre in the camera frame
    if zc <= 0.05:
        return None
    xb, yb = zc + C.CAM_OFFSET[0], -xc          # camera -> base frame (x forward, y left)
    return float(np.hypot(xb, yb)), float(np.arctan2(yb, xb))


#synthetic camera for the kinematic twin
def project_corners(pose, marker):
    #Pixel corners of a marker seen from base pose [x, y, th] and returns (corners_px, view_angle, depth)
    th = pose[2]
    c, s = np.cos(th), np.sin(th)
    cam_w = np.array([pose[0] + C.CAM_OFFSET[0] * c, pose[1] + C.CAM_OFFSET[0] * s, C.CAM_OFFSET[2]])
    P = marker.corners_world() - cam_w # world -> camera origin
    fwd = P[:, 0] * c + P[:, 1] * s
    left = -P[:, 0] * s + P[:, 1] * c
    pc = np.stack([-left, -P[:, 2], fwd], axis=1) # camera frame (OpenCV axes)
    depth = pc[:, 2].mean()
    if np.any(pc[:, 2] <= 0.05):
        return np.full((4, 2), np.nan), np.pi, depth
    uv = np.stack([C.FX * pc[:, 0] / pc[:, 2] + C.CX, C.FY * pc[:, 1] / pc[:, 2] + C.CY], axis=1)
    n = np.array([np.cos(marker.normal), np.sin(marker.normal)])
    to_cam = cam_w[:2] - marker.pos
    view = np.arccos(np.clip(n @ to_cam / max(np.hypot(*to_cam), 1e-9), -1, 1))
    return uv, float(view), float(depth)


def in_image(uv):
    u, v = uv[:, 0], uv[:, 1]
    return bool((u >= 0).all() and (u < C.IMG_W).all() and (v >= 0).all() and (v < C.IMG_H).all())


def synth_measurements(pose, obstacles, rng, dropout=0.0):
    #[(id, range, bearing), ...] the camera would deliver from the true pose.
    # A marker is reported if it faces the camera, lies fully in the image, is within range and large enough, is not occluded by an obstacle, and is not randomly dropped.
    out = []
    pose = np.asarray(pose, float)
    cam_xy = pose[:2] + C.CAM_OFFSET[0] * np.array([np.cos(pose[2]), np.sin(pose[2])])
    for m in MARKERS:
        uv, view, _ = project_corners(pose, m)
        if np.isnan(uv).any() or view > C.CAM_MAX_VIEW_ANGLE or not in_image(uv):
            continue
        if np.hypot(*(m.pos - cam_xy)) > C.CAM_MAX_RANGE or min_side_px(uv) < C.CAM_MIN_SIDE_PX:
            continue
        face = m.pos + 0.01 * np.array([np.cos(m.normal), np.sin(m.normal)])
        if obstacles and segment_blocked(cam_xy, face, obstacles):
            continue
        if dropout and rng.random() < dropout:
            continue
        rb = pnp_range_bearing(uv + rng.normal(0, C.CAM_PIXEL_NOISE, uv.shape), m.size)
        if rb is not None and rb[0] <= C.CAM_MAX_RANGE + 0.3:
            out.append((m.id, *rb))
    return out
