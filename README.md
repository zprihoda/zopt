# zproj
A random collection of small projects

## TODO:
- LQR Dem0
    - Put together simple simulator
        - Replace simulator in lqrControl
        - Use diffrax for ode integration?
    - Write animation / plotting tools
        - Replace plotting tools in lqrControl
    - Consider creating lqrUtil to track common functions
    - Add relevant tests for all of the above (lqrControl, simulator, plottingTools, animationTools, lqrUtil)
    - Add tracking LQR demo and tests
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
