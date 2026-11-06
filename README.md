# Project_69_Track_A

**Group 69**

| Member | Student ID | Contribution |
|---|---|---|
| Christian Wuhrer | 25877919 | `README`, `requirements.txt`, evaluation (`src/eval`), sensor calibration (`scripts/calibrate_sensor.py`). |
| Flynn McDermott | 25840277 | Perception (`src/perception`: ArUco camera pipeline and LiDAR obstacle detection), `scripts/check_setup.py`, `scripts/run_experiments.py`, project page deliverables. |
| Photsatorn Wangsuphadilok | 14710976 | State estimation (`src/estimation`), model analysis (`scripts/analyse_model.py`), repository file structure. |
| Sabina Kazkenova | 14628301 | Planning and control (`src/control`), platform identification (`scripts/identify_platform.py`), simulation (`src/sim`). |

**Repository:** https://github.com/zingerthebox/Project_69_Track_A

**Track:** Track A, mobile robot. A TurtleBot3-style differential-drive robot with a 2D LiDAR and a camera, simulated in MuJoCo. It navigates a room with obstacles and visually docks to an ArUco-marked charging station.
Track A, mobile robot: visual waypoint navigation. TurtleBot3 with 2D LiDAR and camera in MuJoCo. Navigate a room with obstacles using the LiDAR and visually dock to the ArUco-marked charging station.

## Project Overview
 
**Task:** The robot starts at a random pose in a 6 m x 5 m room, avoids three known obstacles and one hidden obstacle (detected with LiDAR and re-planned around), and docks 30 cm in front of ArUco marker 0. It is controlled from its **estimated** pose only.
 
**Pipeline:** `wheel encoders + camera (ArUco -> PnP -> range/bearing) + LiDAR` -> `EKF pose estimate` -> `A* planner` -> `LQR tracker` -> `wheel commands`
 
**Model:** State `x = [x, y, θ]` (pose of the base in the world frame), input `u = (Δs_L, Δs_R)` wheel travel from the encoders. Motion is a mid-point unicycle step with `ds = (Δs_L + Δs_R)/2` and `Δθ = (Δs_R − Δs_L)/b`, where `b = 0.160 m` is the wheel separation. Commands `(v, ω)` map to wheel speeds through `r = 0.033 m`. The robot is a TurtleBot3 Burger (`V_MAX` 0.22 m/s, `W_MAX` 1.5 rad/s, collision radius 0.105 m). All platform constants are in `src/config.py`; the models are in `src/model.py`.
 
**Frames** (defined in `src/config.py`):
 
| Frame | Origin | Axes |
|---|---|---|
| World `W` | Room corner | x east, y north, θ counter-clockwise from +x |
| Base `B` | Wheel-axle centre | x forward, y left |
| Camera `C` | Camera at `[0.07, 0, 0.15]` m in `B` | OpenCV: x right, y down, z forward |
 
**Frame diagram:** <img width="1462" height="1224" alt="WhatsApp Image 2026-11-06 at 1 54 31 PM" src="https://github.com/user-attachments/assets/cbb6bd45-e53e-4871-99c3-57e8e82ba2f8" />
**Two simulation backends share one interface:** `mujoco` (real physics, rendered camera, ray-cast LiDAR; used for demos and the video) and `kinematic` (fast numpy twin, no graphics; used for many-trial studies).

## Demo Video
 <video width="100%" controls>
  <source src="Video Project 4.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
**Link:** (link here)
 
The video is 5 minutes or less and shows:
 
- [ ] The robot completing the navigate-and-dock task (nominal trial).
- [ ] At least one varied or disturbed trial (`--disturb`: 35 % wheel-slip window and a sideways/yaw shove, or a different `--seed`).
- [ ] The estimate and its 2-sigma uncertainty ellipse alongside the true state (the dashboard video draws both).
To regenerate the footage:
 
```
python main.py --backend mujoco --video
python main.py --backend mujoco --disturb --video --tag disturbed
```

## Planning and Control

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

**Results**

| Tracker | Scenario | Cross-track RMSE (cm) | Final dock error (cm / deg) | Time (s) | Replans | Min clearance (cm) | Control effort | Collided | Success |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| LQR | Nominal | 0.96 | 0.23 / 0.29 | 37.2 | 1 | 3.61 | 5.08 | No | Yes |
| PID | Nominal | 1.94 | 0.43 / 0.22 | 36.8 | 1 | 6.85 | 3.85 | No | Yes |
| LQR | Disturbed | 1.29 | 2.35 / 6.15 | 37.95 | 1 | 0.84 | 5.09 | No | No |
| PID | Disturbed | 1.95 | 2.54 / 6.30 | 39.15 | 3 | 10.49 | 5.60 | No | No |

## Estimation and Model Analysis

**Estimators** 
*UKF*
<img width="1801" height="1135" alt="image" src="https://github.com/user-attachments/assets/ece9d6b7-010f-4db6-a9bf-cee45b86420f" />
*Odometry only*
<img width="1804" height="1135" alt="image" src="https://github.com/user-attachments/assets/1cf08c41-fe4b-4f6e-9da4-565ea59c5f84" />
*EKF*
<img width="1801" height="1135" alt="image" src="https://github.com/user-attachments/assets/8181bbc1-f601-4331-a325-83afa333d0d5" />


- **Prediction:** wheel travel `u = (Δs_L, Δs_R)` from the encoders drives a unicycle model. Odometry noise grows with distance travelled: each wheel has variance `ODOM_K·|Δs| + ODOM_FLOOR²`, propagated to the state through the input Jacobian: `Qd = Q_SCALE · G · diag(var) · Gᵀ`.
- **Update:** range and bearing to ArUco markers at known map positions (from camera + PnP). The measurement noise grows with range: `R = R_SCALE · diag[(a₀ + a₂r²)², (b₀ + b₁r)²]`.
- **EKF:** linearised update with the Joseph-form covariance (keeps P symmetric and positive definite) and a wrapped bearing residual.
- **UKF:** 7 sigma points, circular mean for heading and bearing angles.
- **Outlier gating:** an update is rejected when its NIS (normalised innovation squared) exceeds `GATE_CHI2`.
- **`odom`:** dead reckoning only (no updates), used as the baseline.

| Parameter | Value |
|---|---|
| Initial std (position, heading) `P0_XY`, `P0_TH` | [0.1] m, [0.1] rad |
| Odometry noise `ODOM_K`, `ODOM_FLOOR` | [4x10^-4], [1x10^-4] |
| Range noise `RNG_A0`, `RNG_A2` | [0.012], [0.008], [so σ_r = 0.012 + 0.008·r²] |
| Bearing noise `BRG_B0`, `BRG_B1` | [0.004], [0.0005], [so σ_b = 0.004 + 0.0005·r] |
| Tuning scales `Q_SCALE`, `R_SCALE` | [4.0], [1.0] |
| Gate `GATE_CHI2` | [9.21 (99% chi-square threshold for 2 degrees of freedom)] |
| UKF α, β, κ | 0.5, 2.0, 0.0 |

The estimators only receive encoder travel and marker range/bearing. They never receive ground truth.

**Model analysis**
<img width="1920" height="672" alt="image" src="https://github.com/user-attachments/assets/7dd9d0f7-12a0-4ebf-93cb-659d388c10c5" />


- *Controllability* of the linearised tracking-error model [FILL: rank at each reference speed, from the output]. It loses rank at standstill (`v_r = w_r = 0`), and the plot shows the smallest singular value falling to zero as `v_r → 0`. Adding the motor lag gives a 5-state model [FILL: rank, /5].
- *LQR closed loop:* at `v_r = 0.18 m/s` the discrete closed-loop poles are [FILL], with max |z| = [FILL] (< 1 means stable).
- *Observability* of `[x, y, θ]` from camera range-bearing, over a short window:

| Case | Rank | Meaning |
|---|---|---|
| no marker | [0]/3 | No measurements, so only dead reckoning is possible and the covariance grows with distance travelled. |
| 1 marker, static | [2]/3 | One range-bearing fix gives 2 constraints for 3 unknowns, so one direction of the pose is unobservable.|
| 2 markers, static | [3]/3 | Two markers give 4 constraints, so the full pose is observable even without motion.|
| 1 marker, driving an arc | [2]/3 | Driving does not fix the ambiguity|
| 3 markers, driving | [3]/3 | Full rank with the largest singular values, so the best-conditioned case.|


**Results**

| Estimator | Position RMSE (cm) | Final dock error (cm / deg) | Cross-track RMSE (cm) | Mean NIS | Mean NEES | Min clearance (cm) | Collided | Success |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Odometry only | 195.0 | 359.7 / 71.1 | 2.27 | n/a | 199.6 | -0.52 | Yes | No |
| EKF | 2.70 | 2.35 / 6.15 | 1.29 | 0.41 | 28.8 | 0.84 | No | No |
| UKF | 2.94 | 2.43 / 6.09 | 1.27 | 0.39 | 29.4 | 1.45 | No | No |


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




