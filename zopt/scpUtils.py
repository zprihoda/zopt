"""
TODO:
- Control/state bound constraints?
- Replace terminal cost with terminal state?
- Add trust region constraints
- See Stanford HW4:obstacle_avoidance for other options to include
"""

import cvxpy as cvx
import numpy as np

from typing import Callable

from zopt.pytrees import Trajectory, AffineDynamics


def _setupProblem(N, n, m, dt, runningCost, terminalCost, x0):
    x = cvx.Variable((N + 1, n), name="x")
    u = cvx.Variable((N, m), name="u")
    g = lambda x, u: dt * runningCost(x, u)
    gf = terminalCost
    f = cvx.Parameter((N, n), name="f")
    f_x = cvx.Parameter((N, n, n), name="f_x")
    f_u = cvx.Parameter((N, n, m), name="f_u")
    xTraj0 = cvx.Parameter((N, n), name="x0")
    uTraj0 = cvx.Parameter((N, m), name="u0")

    cost = gf(x[-1]) + cvx.sum([g(x[k], u[k]) for k in range(N)])
    objective = cvx.Minimize(cost)

    dyn = lambda x, u, k: x[k] + dt * (f[k] + f_x[k] @ (x[k] - xTraj0[k]) + f_u[k] @ (u[k] - uTraj0[k]))

    constraints = [x[0] == x0]
    constraints += [x[k + 1] == dyn(x, u, k) for k in range(N)]
    prob = cvx.Problem(objective, constraints)
    return prob


def _solveProblem(prob, affine_dynamics):
    # Updates prob parameters and solves
    prob.param_dict['f'].value = np.asarray(affine_dynamics.f)
    prob.param_dict['f_x'].value = np.asarray(affine_dynamics.f_x)
    prob.param_dict['f_u'].value = np.asarray(affine_dynamics.f_u)
    prob.param_dict['x0'].value = np.asarray(affine_dynamics.x0)
    prob.param_dict['u0'].value = np.asarray(affine_dynamics.u0)

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
    tol : Convergence tolerance

    Returns
    -------
    traj : Optimal trajectory
    converged : Whether the algorithm converged
    """
    traj = traj0

    n = traj.xTraj.shape[1]
    N, m = traj.uTraj.shape

    converged = False
    cvxProb = _setupProblem(N, n, m, dt, runningCost, terminalCost, x0)
    J_prev = np.inf
    for i in range(maxIter):
        affine_dynamics = AffineDynamics.from_trajectory(dynamics, traj)
        traj, J = _solveProblem(cvxProb, affine_dynamics)
        if abs(J - J_prev) < tol:
            converged = True
            break
        J_prev = J

    return traj, converged
