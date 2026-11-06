import numpy as np

#platform
WHEEL_RADIUS = 0.033
TRACK = 0.160
BODY_RADIUS = 0.105
V_MAX = 0.22
W_MAX = 1.5
CAM_OFFSET = np.array([0.07, 0.0, 0.15])
LIDAR_Z = 0.14
ENCODER_TICKS = 4096
MOTOR_TAU = 0.05
#time
DT_CTRL = 0.05
CAMERA_EVERY = 2
LIDAR_EVERY = 4
#room, dock, markers
ROOM_W, ROOM_H = 6.0, 5.0
WALL_T = 0.10
DOCK_ID = 0
DOCK_STANDOFF = 0.30
#camera
IMG_W, IMG_H = 640, 480
FOVY_DEG = 60.0
FY = (IMG_H / 2) / np.tan(np.radians(FOVY_DEG) / 2)
FX = FY
CX, CY = IMG_W / 2, IMG_H / 2
K_CAM = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
CAM_MAX_RANGE = 4.0
CAM_MIN_SIDE_PX = 14
CAM_MAX_VIEW_ANGLE = np.radians(70)
CAM_PIXEL_NOISE = 0.5
MARKER_FLIP = "none"
ARUCO_DICT = "DICT_4X4_50"
# lidar
LIDAR_BEAMS = 360
LIDAR_MIN, LIDAR_MAX = 0.12, 3.5
LIDAR_SIGMA0, LIDAR_SIGMA1 = 0.01, 0.005
#est
ODOM_K = 4e-4
ODOM_FLOOR = 1e-4
RNG_A0, RNG_A2 = 0.012, 0.008
BRG_B0, BRG_B1 = 0.004, 0.0005
GATE_CHI2 = 9.21
P0_XY, P0_TH = 0.10, 0.10
Q_SCALE = 4.0
R_SCALE = 1.0
#lqr track
Q_LQR = np.diag([1 / 0.06**2, 1 / 0.03**2, 1 / np.radians(6)**2]) / 8
R_LQR = np.diag([1 / 0.10**2, 1 / 0.8**2]) / 8
#PI track
KP_X, KP_Y, KP_TH = 1.6, 6.0, 3.0
KI_X, KI_Y = 0.15, 0.6
I_MAX = 0.15
# traj lim
V_CRUISE = 0.18
A_LAT = 0.12
A_LONG = 0.25
DS = 0.02
