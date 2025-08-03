# zproj
A random collection of small projects

## TODO:
- Tracking LQR:
    - Plot open loop vs closed loop trajectory
        - Requires updates to plotting tools
    - Run Simulation with wind and/or added noise

- Short Tasks:
    - Create controllers module
        - Move proportionalFeedbackController from lqrUtils to controllers
        - Add integral controller from integralLqrControl demo.
        - Create/move tests

- Tasks:
    - Add iterative LQR demo
    - Add discrete LQR (for all of the above?)
    - Write quadcopter animation

- Simulator:
    - Add support for discrete controllers?
    - Jit the step function? Will that work in a class?
    - Use diffrax for ode integration?
- PlottingTools
    - Add support for structured arrays, can auto-extract names
    - Rename plotTrajectory to plotTrajectorySubplots
    - Add plotTrajectorySingleAx (plots multiple signals on one axis)
    - Add fig/axs optional arguments (plot on the specified figure / axes)
    - Consider general function with
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
