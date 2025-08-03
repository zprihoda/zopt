# zproj
A random collection of small projects

## TODO:
- Integral LQR
    - Generalize plotting tools
        - Only one figure per call, just handle the splitting externally
        - So we can call it like:
        ```
        plotTimeTrajectory(tArr, xArr[:,ind], names)
        plotTimeTrajectory(tArr, xArr[:,ind2], names2)
        plotTimeTrajectory(tArr, xCtrlArr[:,ind2], names3)
        plotTimeTrajectory(tArr, uArr, names4)
        ```
    - Consider removing proportionalFeedbackController from lqrUtils.
        - Move it into the required demos
        - Can eventually create a common controllers module later.
- Tasks:
    - Add tracking LQR demo
    - Add iterative LQR demo
    - Add discrete LQR (for all of the above?)
    - Write quadcopter animation
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
