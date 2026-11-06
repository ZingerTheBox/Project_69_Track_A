# Project_69_Track_A

**Group 69**

| Name | Student ID |

| [Christian Wuhrer] | 25877919 |

| [Flynn McDermott] | 25840277 |

| [Sabina Kazkenova] | 14628301 |

| [Photsatorn Wangsuphadilok] | 14710976 |

**Repository:** https://github.com/zingerthebox/Project_69_Track_A  *(check this link)*

**Track:** Track A, mobile robot. A TurtleBot3-style differential-drive robot with a 2D LiDAR and a camera, simulated in MuJoCo. It navigates a room with obstacles and visually docks to an ArUco-marked charging station.
Track A, mobile robot: visual waypoint navigation. TurtleBot3 with 2D LiDAR and camera in MuJoCo. Navigate a room with obstacles using the LiDAR and visually dock to the ArUco-marked charging station.

## Planning and Control (14628301)

**Planner.** A* on a 0.05 m occupancy grid (obstacles inflated by robot radius + 0.10 m) from the estimated position to a pre-dock point 0.9 m in front of the dock. The path is shortcut by line-of-sight checks and smoothed with Chaikin corner cutting, then given a speed profile limited by curvature (`a_lat`), acceleration (`a_long`), and a slow-down near the dock.

| `v_cruise` | `a_lat` | `a_long` | `ds` |
|---|---|---|---|
| [0.18m/s] | [0.10m/s^2] | [0.05m/s^2] | [0.02m] |

**LQR tracker.** `[v, ω] = [v_ref, ω_ref] − K·e`, with `e` the pose error in the robot frame and `K` gain-scheduled on the reference speeds.
Q = diag([10, 10, 5]), R = diag([1, 1]).

**PID tracker.** Same error `e`, with separate gains on along-track, cross-track and heading error plus clamped integral terms.

| k_x | k_y | k_θ | k_ix | k_iy | integral limit |
|---|---|---|---|---|---|
| 1.6 | 6.0 | 3.0 | 0.15 | 0.6 | 0.15 |

**Simulation backends.** `kinematic` is a fast differential-drive model with motor lag, slip and pushes. `mujoco` is full physics (5 ms step, velocity-actuated wheels, frictionless casters). Platform parameters (motor lag, odometry scale error, slip loss) were measured with `scripts/identify_platform.py`.

**Ground truth.** The backends expose it only through `truth()`, marked evaluation-only. It goes to the logger and metrics, never to the estimator, planner or controller.

**Results (my part).** [table: LQR vs PID tracking error and docking success, nominal vs disturbed]



Install Steps:
    py -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

IF scripts is disabled:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Check if everything is properly installed:
    python -m pytest -q

How to get results:
    main run:
        python main.py #Prints overview, estimation, control .png's
    Video creation:
        python main.py --video --stride 4 #no side camera

    Different Commands for different results:
        python main.py --backend mujoco --video #backend video
        python main.py --backend mujoco --disturb --video --tag disturbed #Disturbed video and results
        python main.py --backend mujoco --viewer #Live View, no speed control
    Different inputs:
        --backend kinematic, --backend mujoco, --estimator efk, --estimator ukf, --estimator odom, --disturb, --viewer, --video, --gif, --stride (1-4), --tracker lqr, --tracker pid




