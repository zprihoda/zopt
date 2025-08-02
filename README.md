# zproj
A random collection of small projects

## TODO:
- Tasks:
    - Write quadcopter animation
    - Add finite horizon LQR demo
    - Add integral LQR demo
    - Add tracking LQR demo
    - Add iterative LQR demo
    - Add discrete LQR (for all of the above?)
- Simulator:
    - Add support for controller states (eg. for integrators)
    - Add support for discrete controllers?
    - Jit the step function? Will that work in a class?
    - Use diffrax for ode integration?
- PlottingTools
    - Add support for structured arrays, can auto-extract names
    - Add plot positionTrajectory (or generic phase plot?)
- Quadcopter:
    - Consider structured arrays (or something similar) to clean up state interface
        - Can then also update other tools to work with structured arrays
    - Speed up trim, consider [optimistix](https://github.com/patrick-kidger/optimistix)
    - Add rotor dynamics
        - Add min/max rpm
    - Add rigid-body / rotor dynamics cross-coupling
        - eg. gyro moments,
    - Add generic dynamics function with options
        - Include inertial dynamics
        - Include rotor dynamics
- Project Config
    - Look into replacing flake8+yapf with ruff
