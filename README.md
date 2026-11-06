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

 **Modelling assumptions** (used by the estimator and controller):

- Planar motion on a flat floor, rigid chassis, pure rolling of the wheels (no lateral slip). Slip and shoves exist only in the simulator as disturbances, so the estimator and controller do not model them.
- Wheel radius and track width are known at their nominal values. The simulator perturbs them slightly, so odometry has a small scale error.
- Marker positions and the three known obstacles are known in the world frame. The hidden obstacle is not.
- Marker measurements are range and bearing with zero-mean Gaussian noise that grows with range.
- 
**Frames** (defined in `src/config.py`):
 
| Frame | Origin | Axes |
|---|---|---|
| World `W` | Room corner | x east, y north, θ counter-clockwise from +x |
| Base `B` | Wheel-axle centre | x forward, y left |
| Camera `C` | Camera at `[0.07, 0, 0.15]` m in `B` | OpenCV: x right, y down, z forward |
 
**Frame diagram:**

<img src="https://github.com/user-attachments/assets/cbb6bd45-e53e-4871-99c3-57e8e82ba2f8" alt="Frame diagram showing the world, base and camera frames" width="600" style="max-width:100%; height:auto;">

**Two simulation backends share one interface:** `mujoco` (real physics, rendered camera, ray-cast LiDAR; used for demos and the video) and `kinematic` (fast numpy twin, no graphics; used for many-trial studies).

## Demo Video
 <video width="100%" controls>
  <source src="Video Project 4.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

The video is 5 minutes or less and shows:
 
- The robot completing the navigate-and-dock task (nominal trial).
- At least one varied or disturbed trial (`--disturb`: 35 % wheel-slip window and a sideways/yaw shove, or a different `--seed`).
- The estimate and its 2-sigma uncertainty ellipse alongside the true state (the dashboard video draws both).
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

Why these values: cruise speed is about 80 % of `V_MAX` (0.22 m/s), which leaves headroom for the tracker's corrections (the commanded speed reaches the 0.22 m/s limit during the re-plan). The 0.10 m inflation margin on top of the robot radius absorbs the 1 to 3 cm tracking and estimation errors seen in the results below.

**Tracking-error model.** The error is expressed in the robot frame, `e = [e_x, e_y, e_θ]` (along-track, cross-track, heading). Linearising about the reference `(v_r, ω_r)` gives `ė = A e + B δu` with `δu = [v − v_r, ω − ω_r]`:

```
A = [[ 0,    ω_r,  0  ],        B = [[-1,  0],
     [-ω_r,  0,    v_r],             [ 0,  0],
     [ 0,    0,    0  ]]             [ 0, -1]]
```

This model is discretised at the control period and used for the LQR and for the controllability analysis below.

**LQR tracker.** `[v, ω] = [v_ref, ω_ref] − K·e`, with `e` the pose error in the robot frame and `K` gain-scheduled on the reference speeds.

`Q = diag(34.7, 138.9, 11.4)`, `R = diag(12.5, 0.195)`

The weights follow Bryson's rule, `1 / (2 · max²)` for the largest acceptable error or input: 0.12 m along-track, 0.06 m cross-track and 12° heading for `Q`; 0.2 m/s and 1.6 rad/s for `R`. The cross-track weight is the largest because a lateral error can only be removed by steering, and the tightest allowed error (6 cm) keeps the robot on the planned path around obstacles. At `v_r = 0.18 m/s` the discrete closed-loop poles are about 0.92, 0.685 and 0.969, so the loop is stable (max |z| ≈ 0.969).

**PI tracker** (`--tracker pid`). A comparison baseline with the same error `e`:

```
v = v_ref·cos(e_θ) + k_x·e_x + k_ix·∫e_x
ω = ω_ref + v_ref·(k_y·e_y + k_iy·∫e_y) + k_θ·sin(e_θ)
```

There is no derivative term, so this is a PI law with a proportional heading term, not a full PID. The integrals are clamped to ±0.15.

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

Cross-track RMSE is measured on the **estimated** pose against the reference.

- **Nominal:** both trackers dock successfully. LQR has about half the cross-track error of the PI tracker (0.96 vs 1.94 cm) and a smaller final position error. The PI tracker uses less control effort (3.85 vs 5.08) and keeps more clearance from obstacles (6.85 vs 3.61 cm).
- **Disturbed:** both trackers fail the dock check in the same way, with about 6° of heading error and 2.4 to 2.5 cm of position error, even though their cross-track errors on the estimated pose are small. The tracker is following its reference well, so the problem is in the pose estimate (see the failure analysis below). The PI tracker needs 3 re-plans against 1 for LQR. LQR passes within 0.84 cm of an obstacle, which is almost a collision, while the PI tracker keeps 10.5 cm.

## Estimation and Model Analysis

- **Prediction:** wheel travel `u = (Δs_L, Δs_R)` from the encoders drives a unicycle model. Odometry noise grows with distance travelled: each wheel has variance `ODOM_K·|Δs| + ODOM_FLOOR²`, propagated to the state through the input Jacobian: `Qd = Q_SCALE · G · diag(var) · Gᵀ`.
- **Update:** range and bearing to ArUco markers at known map positions (from camera + PnP). The measurement noise grows with range: `R = R_SCALE · diag[(a₀ + a₂r²)², (b₀ + b₁r)²]`.
- **EKF:** linearised update with the Joseph-form covariance (keeps P symmetric and positive definite) and a wrapped bearing residual.
- **UKF:** 7 sigma points, circular mean for heading and bearing angles.
- **Outlier gating:** an update is rejected when its NIS (normalised innovation squared) exceeds `GATE_CHI2`.
- **`odom`:** dead reckoning only (no updates), used as the baseline.

| Parameter | Value |
|---|---|
| Initial std (position, heading) `P0_XY`, `P0_TH` | 0.1 m, 0.1 rad |
| Odometry noise `ODOM_K`, `ODOM_FLOOR` | 4×10⁻⁴, 1×10⁻⁴ |
| Range noise `RNG_A0`, `RNG_A2` | 0.012, 0.008 (σ_r = 0.012 + 0.008·r²) |
| Bearing noise `BRG_B0`, `BRG_B1` | 0.004, 0.0005 (σ_b = 0.004 + 0.0005·r) |
| Tuning scales `Q_SCALE`, `R_SCALE` | 4.0, 1.0 |
| Gate `GATE_CHI2` | 9.21 (99 % chi-square threshold for 2 degrees of freedom) |
| UKF α, β, κ | 0.5, 2.0, 0.0 |

**Why these values.**

- **`Q_SCALE = 4`:** odometry is biased as well as noisy. The simulator's true odometry variance (6×10⁻⁴ per metre) is higher than the filter's assumed 4×10⁻⁴, and the wheel radius and track width carry small scale errors. We checked this with a noise sweep (`scripts/run_experiments.py`, `results/fig_noise_sweep.png`: Q and R scaled to 0.25×, 1× and 4× of the configured values, 30 trials per setting). Cutting Q to a quarter, which is an effective `Q_SCALE` of 1, gave 6 to 10 failed trials in 30, against at most 2 in 30 at the configured Q.
- **`R_SCALE = 1`:** the measurement noise comes from calibration (next bullet), and the result is insensitive to it: scaling R to 0.25× or 4× each gave 1 failed trial in 30.
- **Range and bearing noise:** fitted from simulated detections against ground truth with `scripts/calibrate_sensor.py` (`results/sensor_calibration_twin.png`). Range noise grows with r² because the marker's size in pixels shrinks with distance, and bearing noise grows roughly linearly.
- **`GATE_CHI2 = 9.21`:** the 99 % chi-square value for a 2-D innovation. It rejects clear outliers (about 1 % of good updates in expectation). Its cost is discussed in the failure analysis.
- **`P0 = 0.1 m, 0.1 rad`:** the start pose prior is uncertain by about 10 cm and 6°, which matches the covariance at t = 0 in the plots.
- **UKF `α = 0.5, β = 2, κ = 0`:** `β = 2` is the standard choice for Gaussian states, and `α` sets a moderate spread of the sigma points.

The estimators only receive encoder travel and marker range/bearing. They never receive ground truth.

**Estimation plots** (disturbed scenario, one plot per estimator; left: error against the ±3σ band, right: NEES, NIS and the filter's own uncertainty):

*EKF*

<img src="https://github.com/user-attachments/assets/8181bbc1-f601-4331-a325-83afa333d0d5" alt="EKF estimation, disturbed scenario" width="800" style="max-width:100%; height:auto;">

*UKF*

<img src="https://github.com/user-attachments/assets/ece9d6b7-010f-4db6-a9bf-cee45b86420f" alt="UKF estimation, disturbed scenario" width="800" style="max-width:100%; height:auto;">

*Odometry only*

<img src="https://github.com/user-attachments/assets/1cf08c41-fe4b-4f6e-9da4-565ea59c5f84" alt="Odometry-only estimation, disturbed scenario" width="800" style="max-width:100%; height:auto;">

**Reading the consistency numbers.**

- **Nominal:** in the single MuJoCo trials mean NEES is 0.63 to 0.95 (ideal 3) and mean NIS is 0.07 to 0.10 (ideal 2). Both are below ideal, so the filter is **conservative**: its stated uncertainty is larger than its real error. This is the result of the inflated `Q_SCALE`, and it is the safe direction for a filter that gates its own updates. Averaged over 30 kinematic trials the nominal NEES is 2.67, close to the ideal 3, while the mean NIS of 0.70 is still below 2.
- **Disturbed:** mean NEES rises to about 29 in the MuJoCo trial and to 13.8 over 30 kinematic trials (ideal 3), so the filter is **over-confident**. The mean is dominated by two short excursions: the slip window around 15 to 20 s, where NEES peaks near 60 and then recovers, and a jump near the end of the run. Outside those, NEES stays inside the 99 % band. Slip and shoves are not in the filter's model, so it cannot predict errors of that size.
- **Mean NIS stays at about 0.4** in the disturbed run because the gated-out updates are excluded from the average.

**Model analysis**

<img src="https://github.com/user-attachments/assets/7dd9d0f7-12a0-4ebf-93cb-659d388c10c5" alt="Controllability and observability analysis" width="900" style="max-width:100%; height:auto;">

- *Controllability* of the linearised tracking-error model: rank 2/3 at standstill (`v_r = w_r = 0`), where `A = 0` and only the two input directions can be reached. It is rank 3/3 for every moving or turning reference tested: (v_r, w_r) = (0.03, 0), (0.18, 0), (0, 0.5) and (0.18, 0.4). With the first-order motor lag the model has 5 states and is 5/5 controllable at 0.18 m/s. The plot shows the smallest singular value rising linearly with `v_r`, which is why the planner never commands less than 0.03 m/s.
- *LQR closed loop:* at `v_r = 0.18 m/s` the discrete closed-loop poles are about 0.92, 0.685 and 0.969, with max |z| ≈ 0.969 (below 1, so stable).
- *Observability* of `[x, y, θ]` from camera range-bearing, over a short window:

| Case | Rank | Meaning |
|---|---|---|
| no marker | 0/3 | No measurements, so only dead reckoning is possible and the covariance grows with distance travelled. |
| 1 marker, static | 2/3 | One range-bearing fix gives 2 constraints for 3 unknowns, so one direction of the pose is unobservable. |
| 2 markers, static | 3/3 | Two markers give 4 constraints, so the full pose is observable even without motion. |
| 1 marker, driving an arc | 2/3 | Driving does not help: rotating the whole pose and path about the single marker leaves range, bearing and relative odometry unchanged. |
| 3 markers, driving | 3/3 | Full rank with the largest singular values, so the best-conditioned case. |

This is why the estimate degrades whenever fewer than two markers are visible, and why the uncertainty in the plots above grows between sightings and collapses when markers return.

**Results** (MuJoCo backend, seed 3, LQR tracker, one trial per row):

| Scenario | Estimator | Position RMSE (cm) | Final dock error (cm / deg) | Cross-track RMSE (cm) | Mean NIS | Mean NEES | Min clearance (cm) | Collided | Success |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Nominal | EKF | 1.56 | 0.23 / 0.29 | 0.96 | 0.10 | 0.95 | 3.61 | No | Yes |
| Disturbed | Odometry only | 195.0 | 359.7 / 71.1 | 2.27 | n/a | 199.6 | -0.52 | Yes | No |
| Disturbed | EKF | 2.70 | 2.35 / 6.15 | 1.29 | 0.41 | 28.8 | 0.84 | No | No |
| Disturbed | UKF | 2.94 | 2.43 / 6.09 | 1.27 | 0.39 | 29.4 | 1.45 | No | No |

- **Marker updates are essential.** Without them the position error grows to about 2 m RMSE under disturbance, the robot ends 3.6 m from the dock and collides.
- **EKF and UKF perform almost identically** in these single trials (differences of a few tenths of a centimetre). Over 30 nominal trials each they also match: both succeed in 28 of 30, with a mean dock error of 0.34 cm (EKF) and 0.38 cm (UKF). The nominal comparison of all three estimators is in the Repeated Trials section. We do not claim either filter is better.

## Repeated Trials

Single trials are not enough to compare methods, so `scripts/run_experiments.py` repeats the task many times per condition. It ran 420 trials on the kinematic backend (30 trials per condition, 4 workers, 96 s). Per-trial results are in `results/experiments.csv`, failed trials with an automatic diagnosis are in `results/failures.csv`, and the figures are `results/fig_*.png`.

Conditions:

- **Nominal:** LQR tracker, EKF estimator, no disturbance.
- **Disturbed:** nominal plus `--disturb` (wheel-slip window and a shove).
- **Marker dropout 30 % / 60 %:** nominal, with that fraction of marker detections randomly dropped.
- **PI tracker** and **UKF:** nominal with one component swapped.
- **No re-plan:** re-planning disabled, so the hidden obstacle is never avoided.
- **Odometry only:** the `odom` estimator, with no marker updates.

| Condition | Success | Collisions | Mean dock error (cm / deg) | Est. RMSE (cm) | Mean NEES | Mean NIS |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 93.3 % (28/30) | 6.7 % | 0.34 / 0.82 | 4.40 | 2.67 | 0.70 |
| Disturbed | 80.0 % (24/30) | 10.0 % | 1.15 / 1.95 | 4.60 | 13.8 | 0.80 |
| Marker dropout 30 % | 96.7 % (29/30) | 3.3 % | 0.36 / 0.90 | 4.51 | 2.26 | 0.73 |
| Marker dropout 60 % | 93.3 % (28/30) | 6.7 % | 0.44 / 1.23 | 5.12 | 1.89 | 0.76 |
| PI tracker | 93.3 % (28/30) | 6.7 % | 1.81 / 0.61 | 4.56 | 3.02 | 0.69 |
| UKF | 93.3 % (28/30) | 6.7 % | 0.38 / 0.88 | 4.48 | 2.92 | 0.71 |
| No re-plan | 10.0 % (3/30) | 90.0 % | 0.39 / 1.03 | 5.04 | 3.68 | 0.66 |
| Odometry only | 0 % (0/30) | 66.7 % | 105.4 / 21.8 | 66.6 | 2.96 | n/a |

What the 30 trials show:

- **Marker updates are essential:** odometry only never succeeds (0/30), while the EKF succeeds in 28/30. Its position error is 66.6 cm RMSE against 4.4 cm.
- **The LiDAR re-plan is essential:** without it only 3/30 trials succeed and 27/30 collide with the hidden obstacle.
- **Disturbance:** success drops from 93.3 % to 80.0 %, but with 30 trials the 95 % intervals overlap (nominal 79 to 98 %, disturbed 63 to 90 %), so we cannot claim a real drop in success. The mean dock error does grow about threefold (0.34 to 1.15 cm), and NEES rises from 2.7 to 13.8, so the filter becomes over-confident.
- **Marker dropout:** dropping 30 % or 60 % of detections shows no measurable effect on success. At 60 % the position RMSE rises from 4.40 to 5.12 cm.
- **EKF and UKF are equivalent:** the same success rate (28/30) and nearly the same dock error and RMSE.
- **LQR and PI:** the same success rate. LQR docks closer (0.34 against 1.81 cm) and has about half the cross-track error (2.42 against 4.93 cm). The PI tracker has the smaller heading error (0.61° against 0.82°) and uses less control effort (4.07 against 6.35).
- **MuJoCo against kinematic:** the single MuJoCo disturbed trial failed, while the kinematic disturbed condition succeeds 24 times in 30. We do not claim the 80 % figure carries over to MuJoCo.

The noise sweep (Q and R scaled to 0.25×, 1× and 4×, 30 trials per setting) is discussed under the estimator parameters above.

## Limitations

- **The MuJoCo results are single trials** (seed 3). The 30-trial statistics come only from the kinematic twin, and the same conditions were not repeated in MuJoCo.
- **30 trials per condition gives wide intervals.** One trial is 3.3 percentage points, so differences of a few trials (for example 28/30 against 24/30) are not significant.
- **Simulation only.** The sensor noise models were fitted to the simulator, so we make no claim about real hardware.
- **What the results support:** marker updates and the LiDAR re-plan are both necessary for docking; EKF and UKF perform equivalently; LQR docks more accurately than the PI tracker at the same success rate; dropping up to 60 % of marker detections does not measurably reduce success.
- **What they do not support:** any effect of the disturbance on the success rate, any ranking of EKF against UKF, or a transfer of the success rates to MuJoCo or real hardware. In the single disturbed MuJoCo trial both trackers fail for the same estimator-related reason, so that trial says little about the controllers.
- **The estimator is over-confident under disturbance** (mean NEES 13.8 over 30 trials, 28.8 in the MuJoCo trial), and its outlier gate can block genuine corrections, as shown in the failure analysis.
- **The weakest point of the pipeline is the re-plan around the hidden obstacle:** most failures with working estimation are collisions right after it.
- **Model assumptions:** the estimator and controller assume planar motion with no lateral slip and known marker and obstacle positions. LiDAR is used for obstacle detection only, not for localisation.

## Install and Run

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script execution is disabled: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check that everything is installed:

```
python -m pytest -q
```

Run:

```
python main.py                                                      # prints overview, estimation and control PNGs
python main.py --video --stride 4                                   # video, no side camera
python main.py --backend mujoco --video                             # MuJoCo video
python main.py --backend mujoco --disturb --video --tag disturbed   # disturbed video and results
python main.py --backend mujoco --viewer                            # live view, no speed control
```

Options: `--backend kinematic | mujoco`, `--estimator ekf | ukf | odom`, `--tracker lqr | pid`, `--disturb`, `--viewer`, `--video`, `--gif`, `--stride 1-4`, `--tag NAME`.

Other scripts:

```
python scripts/analyse_model.py                       # controllability / observability analysis
python scripts/identify_platform.py --backend mujoco  # motor lag, odometry scale error, slip loss
python scripts/run_experiments.py                     # repeated trials
```

**AI Declaration**

NONE

**Acknowledged sources**

Weekly tutorials sources

https://mujoco.org/

Robotic toolbox


