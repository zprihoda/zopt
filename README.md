# zproj
A random collection of small projects

## TODO:
- Trim and linearization
    - Convert trim to jax.numpy, may have to write my own optimizer
    - Add linearization function
- Dynamics:
    - Add rotor dynamics
        - Add min/max rpm
    - Add rigid-body / rotor dynamics cross-coupling
    - Add gyro moments
    - Add generic dynamics function with options
        - Include inertial dynamics
        - Include rotor dynamics
- General
    - Consider structured arrays (or something similar) to clean up state interface
    - Simple LQR controller
    - Put together simulator
    - Write animation / plotting tools
    - Look into replacing flake8+yapf with ruff
