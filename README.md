# zproj
A random collection of small projects

## TODO:
- LQR Demo
    - Write quadcopter animation
    - Create lqrUtil to track common lqr functions
    - Add relevant tests for all of the above (lqrControl, simulator, plottingTools, animationTools, lqrUtil)
    - Move lqrControl to demo folder, add "long" tests for demos
    - Add integral LQR demo
    - Add tracking LQR demo
    - Add iterative LQR demo
- Simulator:
    - Add support for controller states (eg. for integrators)
    - Add support for discrete controllers?
    - Jit the step function? Will that work in a class?
    - Use diffrax for ode integration?
- PlottingTools
    - Add support for structured arrays, can auto-extract names
    - Add plot positionTrajectory (or generic phase plot?)
- Quadcopter:
    - Speed up trim, consider [optimistix](https://github.com/patrick-kidger/optimistix)
    - Add rotor dynamics
        - Add min/max rpm
    - Add rigid-body / rotor dynamics cross-coupling
        - eg. gyro moments,
    - Add generic dynamics function with options
        - Include inertial dynamics
        - Include rotor dynamics
- General
    - Consider structured arrays (or something similar) to clean up state interface
    - Look into replacing flake8+yapf with ruff
