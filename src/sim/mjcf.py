"""
mjcf.py - builds the MuJoCo scene XML for one Scenario.

Everything comes from config.py and world.py, so MuJoCo and the kinematic twin
always agree on geometry.

Scene: floor, four walls (0.30 m so the 0.14 m LiDAR plane hits them), cylinder
obstacles (orange known ones, red hidden surprise), 9 ArUco marker planes, a
TurtleBot3 Burger (free chassis, two velocity-driven wheels, two frictionless
casters, camera "cam", LiDAR site "lidar") and an overhead camera "top" for the dashboard.

Base frame is x forward, y left, origin at the axle centre. Camera sits at the same
height as the markers (15 cm), LiDAR plane at 14 cm.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .. import config as C
from ..perception.aruco import marker_image
from ..world import MARKER_Z, MARKERS, Scenario, dock_pose

AXLE_Z = C.WHEEL_RADIUS + 0.0005  # base origin height above floor at t=0
WHEEL_KV = 0.02  # velocity actuator gain, gives tau ~ 50 ms (see identify_platform.py)
WALL_H = 0.30
OBST_H = 0.40
MARKER_PLANE_HALF = 0.1  # 0.15 m black square + one 0.025 m white cell each side
CHASSIS_MASS = 0.9  # kg, Burger is ~1 kg total

FLIPS = ["none", "ud", "lr", "both"]


def _f(*v):
    return " ".join(f"{x:.5f}" for x in v)


def write_marker_pngs(asset_dir, flip):
    asset_dir.mkdir(parents=True, exist_ok=True)
    for m in MARKERS:
        g = marker_image(m.id, flip)
        cv2.imwrite(str(asset_dir / f"aruco_{m.id}.png"), cv2.merge([g, g, g]))  # RGB png, safest for MuJoCo


def build_xml(sc, asset_dir, flip=None):
    flip = flip or C.MARKER_FLIP
    write_marker_pngs(asset_dir, flip)
    W, H, T = C.ROOM_W, C.ROOM_H, C.WALL_T
    x0, y0, th0 = sc.start
    cam_h = MARKER_Z - AXLE_Z + 0.0005  # camera height in base frame
    lid_h = C.LIDAR_Z - AXLE_Z + 0.0005
    L2 = C.TRACK / 2

    assets = ['<texture name="floor" type="2d" builtin="checker" rgb1=".80 .82 .85" rgb2=".74 .76 .80" '
              'width="256" height="256"/>',
              '<material name="floor" texture="floor" texrepeat="12 10" texuniform="false" reflectance="0"/>']
    for m in MARKERS:
        assets.append(f'<texture name="tex_m{m.id}" type="2d" file="aruco_{m.id}.png"/>')
        assets.append(f'<material name="mat_m{m.id}" texture="tex_m{m.id}" texuniform="false" '
                      f'texrepeat="1 1" specular="0" shininess="0" reflectance="0"/>')

    world = [f'<geom name="floor" type="plane" pos="{_f(W / 2, H / 2, 0)}" size="{_f(W / 2 + T, H / 2 + T, 0.1)}" material="floor"/>']
    wall_rgba = "0.62 0.62 0.66 1"
    for name, pos, size in (("wall_s", (W / 2, -T / 2), (W / 2 + T, T / 2)), ("wall_n", (W / 2, H + T / 2), (W / 2 + T, T / 2)),
                            ("wall_w", (-T / 2, H / 2), (T / 2, H / 2)), ("wall_e", (W + T / 2, H / 2), (T / 2, H / 2))):
        world.append(f'<geom name="{name}" type="box" pos="{_f(*pos, WALL_H / 2)}" size="{_f(*size, WALL_H / 2)}" rgba="{wall_rgba}"/>')
    for i, (cx, cy, r) in enumerate(sc.obstacles):
        world.append(f'<geom name="obstacle_{i}" type="cylinder" pos="{_f(cx, cy, OBST_H / 2)}" size="{_f(r, OBST_H / 2)}" '
                     f'rgba="0.95 0.55 0.15 1"/>')
    if sc.surprise:
        cx, cy, r = sc.surprise
        world.append(f'<geom name="surprise" type="cylinder" pos="{_f(cx, cy, OBST_H / 2)}" size="{_f(r, OBST_H / 2)}" '
                     f'rgba="0.85 0.15 0.15 1"/>')
    for m in MARKERS:
        n = np.array([np.cos(m.normal), np.sin(m.normal)])
        right, up = np.array([-n[1], n[0], 0.0]), np.array([0.0, 0.0, 1.0])  # viewer's right and up
        world.append(f'<geom name="marker_{m.id}" type="plane" pos="{_f(m.x, m.y, MARKER_Z)}" '
                     f'size="{_f(MARKER_PLANE_HALF, MARKER_PLANE_HALF, 0.001)}" xyaxes="{_f(*right, *up)}" '
                     f'material="mat_m{m.id}" contype="0" conaffinity="0" group="1"/>')
    gx, gy, _ = dock_pose()
    world.append(f'<geom name="dock_pad" type="cylinder" pos="{_f(gx, gy, 0.0006)}" size="0.13 0.0005" '
                 f'rgba="0.2 0.75 0.3 0.5" contype="0" conaffinity="0" group="1"/>')

    robot = f'''
    <body name="base" pos="{_f(x0, y0, AXLE_Z)}" quat="{_f(np.cos(th0 / 2), 0, 0, np.sin(th0 / 2))}">
      <freejoint name="root"/>
      <geom name="chassis" type="cylinder" size="{_f(C.BODY_RADIUS, 0.07)}" pos="0 0 0.07" mass="{CHASSIS_MASS}"
            rgba="0.18 0.20 0.26 1" group="2"/>
      <geom name="lds_puck" type="cylinder" size="0.035 0.02" pos="{_f(0, 0, lid_h)}" mass="0.02" rgba="0.1 0.1 0.1 1"
            contype="0" conaffinity="0" group="2"/>
      <geom name="cam_box" type="box" size="0.01 0.03 0.015" pos="{_f(C.CAM_OFFSET[0] - 0.01, 0, cam_h)}" mass="0.01"
            rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0" group="2"/>
      <!-- priority=1 because MuJoCo mixes friction by max, so the casters would otherwise inherit the floor's mu=1 -->
      <geom name="caster_f" type="sphere" size="0.008" pos="0.09 0 {-(C.WHEEL_RADIUS - 0.008):.5f}" mass="0.01"
            friction="0 0 0" condim="1" priority="1" group="2"/>
      <geom name="caster_b" type="sphere" size="0.008" pos="-0.09 0 {-(C.WHEEL_RADIUS - 0.008):.5f}" mass="0.01"
            friction="0 0 0" condim="1" priority="1" group="2"/>
      <body name="wheel_l" pos="0 {L2:.5f} 0">
        <joint name="wl" type="hinge" axis="0 1 0" damping="0.0002"/>
        <geom name="wheel_l_geom" type="cylinder" size="{C.WHEEL_RADIUS} 0.009" euler="1.5708 0 0" mass="0.03"
              friction="1.0 0.005 0.0001" condim="3" priority="1" rgba="0.1 0.1 0.1 1" group="2"/>
      </body>
      <body name="wheel_r" pos="0 {-L2:.5f} 0">
        <joint name="wr" type="hinge" axis="0 1 0" damping="0.0002"/>
        <geom name="wheel_r_geom" type="cylinder" size="{C.WHEEL_RADIUS} 0.009" euler="1.5708 0 0" mass="0.03"
              friction="1.0 0.005 0.0001" condim="3" priority="1" rgba="0.1 0.1 0.1 1" group="2"/>
      </body>
      <site name="lidar" pos="{_f(0, 0, lid_h)}" size="0.005"/>
      <camera name="cam" pos="{_f(C.CAM_OFFSET[0], 0, cam_h)}" xyaxes="0 -1 0 0 0 1" fovy="{C.FOVY_DEG}"/>
    </body>'''

    return f'''<mujoco model="track_a_docking">
  <compiler angle="radian" texturedir="{asset_dir.as_posix()}" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>   <!-- implicit because of the stiff velocity actuators on light wheels -->
  <visual>
    <global offwidth="{C.IMG_W}" offheight="{C.IMG_H}"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
    <quality shadowsize="0"/>
  </visual>
  <asset>
    {chr(10).join("    " + a for a in assets)}
  </asset>
  <worldbody>
    <light pos="{_f(W / 2, H / 2, 5)}" dir="0 0 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <camera name="top" pos="{_f(W / 2, H / 2, 6.6)}" xyaxes="1 0 0 0 1 0" fovy="45"/>
    {chr(10).join("    " + g for g in world)}
    {robot}
  </worldbody>
  <actuator>
    <velocity name="act_l" joint="wl" kv="{WHEEL_KV}" ctrlrange="-12 12"/>
    <velocity name="act_r" joint="wr" kv="{WHEEL_KV}" ctrlrange="-12 12"/>
  </actuator>
</mujoco>
'''
