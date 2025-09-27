"""
TODO:
- Cleanup general inequality and equality constraints
    - Consider making constraints time/frame dependent (which would also support terminal constraints)
- Add trust region constraints, eg. `npl.norm(xTrajPrev - xPrev) <= rho_x`
- See Stanford HW4:obstacle_avoidance for other options to include
"""

import cvxpy as cvx
import jax.numpy as jnp
import numpy as np

from typing import Callable

from zopt.pytrees import Trajectory, AffineDynamics


class OptimalControlProblem():

    def __init__(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        runningCost: Callable[[np.ndarray, np.ndarray], float],
        terminalCost: Callable[[np.ndarray], float],
        x0: np.ndarray,
        tf: float,
        n_u: int,
        ineqConstraints: Callable[[np.ndarray, np.ndarray], jnp.ndarray] = lambda x, u: jnp.array([]),
        eqConstraints: Callable[[np.ndarray, np.ndarray], jnp.ndarray] = lambda x, u: jnp.array([])
    ):
        """
        Optimal Control Problem

        Arguments
        ---------
        dynamics : Dynamics function of the form: `xDot = f(x,u)`
        runningCost : Running cost function of the form `j = runningCost(x,u)`
        terminalCost : Terminal cost function of the form `jf = terminalCost(xf)`
        x0 : Initial state
        tf : Final time
        n_u : Dimension of control input
        ineqConstraints : Optional inequality constraints of the form: `ineqConstraints(x,u) <= 0`
        eqConstraints : Optional equality constraints of the form: `eqConstraints(x,u) == 0`
        """
        self.dynamics = dynamics
        self.runningCost = runningCost
        self.terminalCost = terminalCost
        self.ineqConstraints = ineqConstraints
        self.eqConstraints = eqConstraints

        self.x0 = x0
        self.tf = tf

        n_x = len(x0)
        self.n_x = n_x
        self.n_u = n_u
        self.n_ineq = ineqConstraints(np.zeros(n_x), np.zeros(n_u)).shape[0]
        self.n_eq = eqConstraints(np.zeros(n_x), np.zeros(n_u)).shape[0]

    def linearizeAboutTrajectory(self, traj: Trajectory, dt: float = 0):
        affine_dynamics = AffineDynamics.from_trajectory(self.dynamics, traj)
        affine_ineq = AffineDynamics.from_trajectory(self.ineqConstraints, traj)
        affine_eq = AffineDynamics.from_trajectory(self.eqConstraints, traj)

        if dt != 0:
            affine_dynamics = AffineDynamics(
                dt * affine_dynamics.f0, np.eye(self.n_x) + dt * affine_dynamics.f_x, dt * affine_dynamics.f_u
            )
        return (affine_dynamics, affine_ineq, affine_eq)


def _setupProblem(N, dt, OCP):
    """
    Setup the cvx problem with tuneable parameters

    NOTE: The parameters are flattened from 3D to 2D arrays to be compatible with cvxpy's canonicalization backend which
        doesn't support > 2D variables / parameters. This will be fixed in a future release of cvxpy, in which case we
        can un-flatten the parameters.
    """
    n = OCP.n_x
    m = OCP.n_u
    n_ineq = OCP.n_ineq
    n_eq = OCP.n_eq

    x = cvx.Variable((N + 1, n), name="x")
    u = cvx.Variable((N, m), name="u")

    f0 = cvx.Parameter((N, n), name="f0")
    f_x = cvx.Parameter((N, n * n), name="f_x")
    f_u = cvx.Parameter((N, n * m), name="f_u")

    dyn = lambda x, u, k: f0[k] + f_x[k].reshape((n, n), 'C') @ x[k] + f_u[k].reshape((n, m), 'C') @ u[k]
    cost = OCP.terminalCost(x[-1]) + dt * cvx.sum([OCP.runningCost(x[k], u[k]) for k in range(N)])
    objective = cvx.Minimize(cost)
    constraints = [x[0] == OCP.x0]
    constraints += [x[k + 1] == dyn(x, u, k) for k in range(N)]

    # Add additional constraints
    if n_ineq > 0:
        f0_ineq = cvx.Parameter((N, n_ineq), name="f0_ineq")
        f_x_ineq = cvx.Parameter((N, n_ineq * n), name="f_x_ineq")
        f_u_ineq = cvx.Parameter((N, n_ineq * m), name="f_u_ineq")
        ineq_fun = lambda x, u, k: f0_ineq[k] + f_x_ineq[k].reshape((n_ineq, n), 'C') @ x[k] + f_u_ineq[k].reshape(
            (n_ineq, m), 'C'
        ) @ u[k]
        constraints += [ineq_fun(x, u, k) <= 0 for k in range(N)]

    if n_eq > 0:
        f0_eq = cvx.Parameter((N, n_eq), name="f0_eq")
        f_x_eq = cvx.Parameter((N, n_eq * n), name="f_x_eq")
        f_u_eq = cvx.Parameter((N, n_eq * m), name="f_u_eq")
        eq_fun = lambda x, u, k: f0_eq[k] + f_x_eq[k].reshape((n_eq, n), 'C') @ x[k] + f_u_eq[k].reshape(
            (n_eq, m), 'C'
        ) @ u[k]
        constraints += [eq_fun(x, u, k) == 0 for k in range(N)]

    prob = cvx.Problem(objective, constraints)
    return prob


def _solveProblem(prob, affine_dynamics, affine_ineq, affine_eq):
    # Updates prob parameters and solves
    N, n, m = affine_dynamics.f_u.shape
    prob.param_dict['f0'].value = np.asarray(affine_dynamics.f0)
    prob.param_dict['f_x'].value = np.asarray(affine_dynamics.f_x).reshape((N, -1))
    prob.param_dict['f_u'].value = np.asarray(affine_dynamics.f_u).reshape((N, -1))

    if affine_ineq.f0.shape[1] > 0:
        prob.param_dict['f0_ineq'].value = np.asarray(affine_ineq.f0)
        prob.param_dict['f_x_ineq'].value = np.asarray(affine_ineq.f_x).reshape((N, -1))
        prob.param_dict['f_u_ineq'].value = np.asarray(affine_ineq.f_u).reshape((N, -1))

    if affine_eq.f0.shape[1] > 0:
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
    ineqConstraints: Callable[[np.ndarray, np.ndarray], jnp.ndarray] = lambda x, u: jnp.array([]),
    eqConstraints: Callable[[np.ndarray, np.ndarray], jnp.ndarray] = lambda x, u: jnp.array([]),
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
    N, m = traj.uTraj.shape
    OCP = OptimalControlProblem(
        dynamics,
        runningCost,
        terminalCost,
        x0,
        N * dt,
        m,
        ineqConstraints=ineqConstraints,
        eqConstraints=eqConstraints
    )

    converged = False
    cvxProb = _setupProblem(N, dt, OCP)
    J_prev = np.inf
    for iter in range(maxIter):
        (affine_dynamics, affine_ineq, affine_eq) = OCP.linearizeAboutTrajectory(traj, dt=dt)
        traj, J = _solveProblem(cvxProb, affine_dynamics, affine_ineq, affine_eq)

        if abs(J - J_prev) < tol:
            converged = True
            break

        J_prev = J

    return traj, converged
