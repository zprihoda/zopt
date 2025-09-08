# zproj
A random collection of small projects

## TODO:
### ilqr-jax
- Implement DDP
- More documentation
    - Add references (Stanford and Paper)
- Consider updating functions that take (x0,u0) and convert to use Trajectory?
    - May make more sense if we rename Trajectory to StateActionPair? But is that better than current approach? Seems confusing?

### All
- ilqr-jax:
    - Update iLQR to be fully jittable, see LQR Variants from stanford examples for help
        - Migrate from class based to function based.
            - More aligned with jax, see: https://docs.jax.dev/en/latest/jep/18137-numpy-scipy-scope.html#axis-5-functional-vs-object-oriented-apis
            - Can do one function at a time
        - Implement each individual function in a jax compliant manner:
            - backward_riccati_step
            - computeQ, consider co
            - Use jax.lax.scan to implement
    - Update other lqr functions to be jittable:
        - Consider jax based implementations of infinite horizon lqr (continuous and discrete)
        - For infinite horizon, compare runtime of scipy vs jitted versions
    - Move cost function types to common modules?
        - Only if it's useful for reuse, eg. Is it useful in other lqr functions?
- Tasks:
    - Indirect methods (port / cleanup Kirk algorithms)
        - Implement fully jax compliant versions of each method?
    - Direct Methods
        - Same as indirect methods?
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
