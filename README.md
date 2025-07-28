# zproj
A random collection of small projects

## TODO:
- Dynamics:
    - Add rotor dynamics
        - Add min/max rpm
    - Add gyro moments
    - Add rigid-body / rotor dynamics cross-coupling
    - Add generic dynamics function with options
        - Include inertial dynamics
        - Include rotor dynamics
- Profile and speed up trim function
    - jit the dynamics function, consider passing in analytical jacaobians
- Add linearization function
- Consider structured arrays (or something similar) to clean up state interface
- Simple LQR controller
- Put together simulator
- Write animation / plotting tools
- Look into replacing flake8+yapf with ruff
