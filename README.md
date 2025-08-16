# zproj
A random collection of small projects

## TODO:
### All
- Animation
    - Plot options:
        - Add 2d traces (xy, yz, zx traces)
    - Add animation to one of the demos (iLQR?)

- Tasks:
    - Indirect methods (port / cleanup Kirk algorithms)
    - Direct Methods
    - MPC
- Short Tasks:
    - Create controllers module?
        - Move proportionalFeedbackController from lqrUtils to controllers
        - Add integral controller from integralLqrControl demo.
        - Create/move tests
- iLQR/DDP:
    - Look into using jax.lax.cond to jit functions with conditionals
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
