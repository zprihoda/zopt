# zproj
A random collection of small projects

## TODO:
### All
- Iterative LQR
    - iLQR + DDP improvements outlined in "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
        - II.C: Improved Value Update
        - II.D: Improved Linear Search
        - II.F: Regularization Schedule
    - Consider adding fullOutput option to iLqr and DDP, so we can plot animations of trajectory over time?
    - Implement an MPC like demo with iLQR
- Short Tasks:
    - Create controllers module
        - Move proportionalFeedbackController from lqrUtils to controllers
        - Add integral controller from integralLqrControl demo.
        - Create/move tests
- Tasks:
    - Write quadcopter animation
    - Indirect methods (port / cleanup Kirk algorithms)
    - Direct Methods
    - MPC

- Simulator:
    - Add support for multi-rate systems
    - Generalize simBlocks + add signal map
    - Write custom ode solver that outputs dx and y, so we can capture outputs as we run
    - Add support for output memory (can implement as a discrete memory block once multi-rate implemented)
        - Would let us skip the resampling after running the sim
    - Consider diffrax for ode integration?
- PlottingTools
    - Add support for structured arrays, can auto-extract names
    - Rename plotTrajectory to plotTrajectorySubplots?
    - Add plotTrajectorySingleAx (plots multiple signals on one axis)
    - Add plot positionTrajectory (or generic phase plot?)
- Quadcopter:
    - Consider structured arrays (or something similar) to clean up state interface
        - Can then also update other tools to work with structured arrays
    - Consider [optimistix](https://github.com/patrick-kidger/optimistix) for trim
    - Add rotor dynamics
        - Add min/max rpm
    - Add rigid-body / rotor dynamics cross-coupling
        - eg. gyro moments,
    - Add generic dynamics function with options
        - Include inertial dynamics
        - Include rotor dynamics
- Project Config
    - Look into replacing flake8+yapf with ruff
    - Document flake8 ignore list
