# Project_69_Track_A

**Group 69**

| Name | Student ID |
|---|---|
| [FILL teammate] | [FILL] |
| [FILL, your name] | 14628301 |

**Repository:** https://github.com/zingerthebox/Project_69_Track_A  *(check this link)*

**Track:** Track A, mobile robot. A TurtleBot3-style differential-drive robot with a 2D LiDAR and a camera, simulated in MuJoCo. It navigates a room with obstacles and visually docks to an ArUco-marked charging station.
Track A, mobile robot: visual waypoint navigation. TurtleBot3 with 2D LiDAR and camera in MuJoCo. Navigate a room with obstacles using the LiDAR and visually dock to the ArUco-marked charging station.





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




