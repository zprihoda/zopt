"""
TODO:
- Cleanup general inequality and equality constraints
    - Create a LOCP pytree object with "from_trajectory" function for linearization
    - Also add a discretization function and cleanup dynamics in cvx problem
        - x + dt*(f + f_x @ x + f_u @ u) = dt*f + (dt*f_x + I) @ x + dt*f_u @ u
    - Consider making constraints time/frame dependent (which would also support terminal constraints)
- Add trust region constraints, eg. `npl.norm(xTrajPrev - xPrev) <= rho_x`
- See Stanford HW4:obstacle_avoidance for other options to include
"""

import cvxpy as cvx
import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable

from zopt.pytrees import Trajectory, AffineDynamics


def _setupProblem(N, n, m, dt, runningCost, terminalCost, x0, n_ineq, n_eq):
    """
    Setup the cvx problem with tuneable parameters

    NOTE: The parameters are flattened from 3D to 2D arrays to be compatible with cvxpy's canonicalization backend which
        doesn't support > 2D variables / parameters. This will be fixed in a future release of cvxpy, in which case we
        can un-flatten the parameters.
    """
    x = cvx.Variable((N + 1, n), name="x")
    u = cvx.Variable((N, m), name="u")
    g = lambda x, u: dt * runningCost(x, u)
    gf = terminalCost

    f0 = cvx.Parameter((N, n), name="f0")
    f_x = cvx.Parameter((N, n*n), name="f_x")
    f_u = cvx.Parameter((N, n*m), name="f_u")

    cost = gf(x[-1]) + cvx.sum([g(x[k], u[k]) for k in range(N)])
    objective = cvx.Minimize(cost)
    dyn = lambda x, u, k: x[k] + dt * (f0[k] + f_x[k].reshape((n,n), 'C') @ x[k] + f_u[k].reshape((n,m), 'C') @ u[k])
    constraints = [x[0] == x0]
    constraints += [x[k + 1] == dyn(x, u, k) for k in range(N)]

    # Add additional constraints
    if n_ineq > 0:
        f0_ineq = cvx.Parameter((N, n_ineq), name="f0_ineq")
        f_x_ineq = cvx.Parameter((N, n_ineq*n), name="f_x_ineq")
        f_u_ineq = cvx.Parameter((N, n_ineq*m), name="f_u_ineq")
        constraints += [f0_ineq[k] + f_x_ineq[k].reshape((n_ineq,n), 'C') @ x[k] + f_u_ineq[k].reshape((n_ineq,m), 'C') @ u[k] <= 0 for k in range(N)]

    if n_eq > 0:
        f0_eq = cvx.Parameter((N, n_eq), name="f0_eq")
        f_x_eq = cvx.Parameter((N, n_eq*n), name="f_x_eq")
        f_u_eq = cvx.Parameter((N, n_eq*m), name="f_u_eq")
        constraints += [f0_eq[k] + f_x_eq[k].reshape((n_eq,n), 'C') @ x[k] + f_u_eq[k].reshape((n_eq,m), 'C') @ u[k] == 0 for k in range(N)]

    prob = cvx.Problem(objective, constraints)
    return prob


def _solveProblem(prob, affine_dynamics, affine_ineq, affine_eq, n_ineq, n_eq):
    # Updates prob parameters and solves
    N,n,m = affine_dynamics.f_u.shape
    prob.param_dict['f0'].value = np.asarray(affine_dynamics.f0)
    prob.param_dict['f_x'].value = np.asarray(affine_dynamics.f_x).reshape((N,-1))
    prob.param_dict['f_u'].value = np.asarray(affine_dynamics.f_u).reshape((N,-1))

    if n_ineq > 0:
        prob.param_dict['f0_ineq'].value = np.asarray(affine_ineq.f0)
        prob.param_dict['f_x_ineq'].value = np.asarray(affine_ineq.f_x).reshape((N, -1))
        prob.param_dict['f_u_ineq'].value = np.asarray(affine_ineq.f_u).reshape((N, -1))

    if n_eq > 0:
        prob.param_dict['f0_eq'].value = np.asarray(affine_eq.f0)
        prob.param_dict['f_x_eq'].value = np.asarray(affine_eq.f_x).reshape((N, -1))
        prob.param_dict['f_u_eq'].value = np.asarray(affine_eq.f_u).reshape((N, -1))

    J = prob.solve()

    xTraj = prob.var_dict['x'].value
    uTraj = prob.var_dict['u'].value

    return Trajectory(xTraj, uTraj), J


def sequentialConvexProgramming(
    traj0: Trajectory,
    x0: np.ndarray,
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
    runningCost: Callable[[np.ndarray, np.ndarray], float],
    terminalCost: Callable[[np.ndarray], float],
    dt: float,
    ineqConstraints: Callable[[np.ndarray, np.ndarray], jnp.ndarray] = lambda x,u: jnp.array([]),
    eqConstraints: Callable[[np.ndarray, np.ndarray], jnp.ndarray] = lambda x,u: jnp.array([]),
    tol: float = 1e-3,
    maxIter=100
) -> tuple[Trajectory, bool]:
    """
    Generate an optimal trajectory via sequential convex programming

    Arguments
    ---------
    traj0 : Initial guess for starting trajectory. Must have shapes:
        traj0.xTraj.shape = (N+1,n)
        traj0.uTraj.shape = (N,m)
    x0 : Initial state. Array of shape (n,)
    dynamics : Continuous dynamics function of the form: `xDot = dynamics(x,u)`
    runningCost : Running cost function of the form `j = runningCost(x,u)`
    terminalCost : Terminal cost function of the form `jf = terminalCost(xf)`
    dt : Discretization time step
    ineqConstraints : Inequality constraint function of the form: `f(x,u) <= 0`
    eqConstraints : Equality constraints of the form: `h(x,u) = 0`
    tol : Convergence tolerance
    maxIter : maximum number of SCP iterations to run

    Returns
    -------
    traj : Optimal trajectory
    converged : Whether the algorithm converged
    """
    traj = traj0

    n = traj.xTraj.shape[1]
    N, m = traj.uTraj.shape

    n_ineq = ineqConstraints(np.zeros(n), np.zeros(m)).shape[0]
    n_eq = eqConstraints(np.zeros(n), np.zeros(m)).shape[0]

    converged = False
    cvxProb = _setupProblem(N, n, m, dt, runningCost, terminalCost, x0, n_ineq, n_eq)
    J_prev = np.inf
    for i in range(maxIter):
        affine_dynamics = AffineDynamics.from_trajectory(dynamics, traj)
        affine_ineq = AffineDynamics.from_trajectory(ineqConstraints, traj)
        affine_eq = AffineDynamics.from_trajectory(eqConstraints, traj)

        traj, J = _solveProblem(cvxProb, affine_dynamics, affine_ineq, affine_eq, n_ineq, n_eq)

        if abs(J - J_prev) < tol:
            converged = True
            break
        J_prev = J

    return traj, converged
