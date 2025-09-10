import jax
import jax.numpy as jnp

from functools import partial
from typing import Callable
from zProj.pytrees import (
    Trajectory,
    AffineDynamics,
    QuadraticDynamics,
    AffinePolicy,
    QuadraticCostFunction,
    QuadraticValueFunction,
    CostFunction,
    QuadraticDeltaCost
)


def trajectoryRollout(
    x0: jnp.ndarray,
    dynFun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    policy: AffinePolicy,
    trajPrev: Trajectory,
    alpha: float = 1
) -> Trajectory:
    """
    Rollout a trajectory from the initial state using the provided control policy

    Arguments
    ---------
        x0 : Initial state
        dynFun : Non-linear dynamics function of the form: `xOut = f(x,u)`
        policy : Affine control policy
        trajPrev : Previous trajectory
        alpha : Step size

    Returns
    -------
        Resulting trajectory
    """
    xPrev, uPrev = trajPrev
    N = uPrev.shape[0]

    def trajectoryStep(x, k):
        dx = x - xPrev[k]
        u = policy(dx, k=k, alpha=alpha) + uPrev[k]
        xOut = dynFun(x, u)
        return xOut, (xOut, u)

    _, (xTraj, uTraj) = jax.lax.scan(trajectoryStep, x0, jnp.arange(N))
    xTraj = jnp.concatenate([x0[None, :], xTraj])
    return Trajectory(xTraj, uTraj)


def forwardPass(
    x0: jnp.ndarray,
    dynFun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    costFun: CostFunction,
    policy: AffinePolicy,
    trajPrev: Trajectory,
    dJFun: QuadraticDeltaCost,
    JPrev: float,
    cLineSearch: float = 0.5,
    alphaMin: float = 0.5**16
) -> tuple[Trajectory, float]:
    """
    ILQR forward pass using the expected quadratic cost change to determine the step size

    Arguments
    ---------
        x0 : Initial state
        dynFun : Non-linear dynamics function of the form: `xOut = f(x,u)`
        costFun : Cost function for step-size line search
        policy : Affine control policy
        trajPrev : Previous trajectorys
        dJFun : Quadratic cost change function: `dJ_exp = djFun(alpha)`
        JPrev : Previous cost
        cLineSearch : Termination criteria for line search: `(J-J_prev) > cLineSearch * dJ_exp`
            Must be between [0,1]
        alphaMin : Minimum step size for line search

    Returns
    -------
        New trajectory and cost
    """

    def forwardPassStep(loopVars):
        J, traj, alpha = loopVars
        trajNew = trajectoryRollout(x0, dynFun, policy, trajPrev, alpha=alpha)
        JNew = costFun(trajNew)
        alpha = alpha * 0.5
        return (JNew, trajNew, alpha)

    def whileCond(loopVars):
        J, traj, alpha = loopVars
        return ((J - JPrev) / dJFun(alpha) <= cLineSearch) | (alpha <= alphaMin)

    J, traj, alpha = jax.lax.while_loop(whileCond, forwardPassStep, (JPrev, trajPrev, 1.))
    return traj, J


def forwardPass2(
    x0: jnp.ndarray,
    dynFun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    costFun: CostFunction,
    policy: AffinePolicy,
    trajPrev: Trajectory
) -> tuple[Trajectory, float]:
    """
    Simplified ILQR forward pass.
    Rollout trajectories for a fixed set of step sizes and select the one with minimum cost.

    Arguments
    ---------
        x0 : Initial state
        dynFun : Non-linear dynamics function of the form: `xOut = f(x,u)`
        costFun : Cost function for step-size line search
        policy : Affine control policy
        trajPrev : Previous trajectorys

    Returns
    -------
        New trajectory and cost
    """

    def forwardPassInner(alpha):
        trajNew = trajectoryRollout(x0, dynFun, policy, trajPrev, alpha=alpha)
        JNew = costFun(trajNew)
        return (JNew, trajNew)

    alphaArr = 0.5**jnp.arange(16)
    (JArr, trajArr) = jax.vmap(forwardPassInner)(alphaArr)
    idx = jnp.argmin(JArr)
    traj = trajArr[idx]
    J = JArr[idx]
    return traj, J


def riccatiStep_ilqr(dynamics: AffineDynamics, cost: QuadraticCostFunction,
                     value: QuadraticValueFunction) -> tuple[QuadraticValueFunction, AffinePolicy]:
    """Perform one step of the backwards Ricatti recursion"""
    _, f_x, f_u = dynamics
    c, c_x, c_u, c_xx, c_ux, c_uu = cost
    v, v_x, v_xx = value

    Q = c + v
    Q_x = c_x + f_x.T @ v_x
    Q_u = c_u + f_u.T @ v_x
    Q_xx = c_xx + f_x.T @ v_xx @ f_x
    Q_uu = c_uu + f_u.T @ v_xx @ f_u
    Q_ux = c_ux + f_u.T @ v_xx @ f_x

    l = -jnp.linalg.solve(Q_uu, Q_u)
    L = -jnp.linalg.solve(Q_uu, Q_ux)

    valueOut = QuadraticValueFunction(Q - 0.5 * l.T @ Q_uu @ l, Q_x - L.T @ Q_uu @ l, Q_xx - L.T @ Q_uu @ L)

    policy = AffinePolicy(l, L)
    return valueOut, policy


def backwardPass_ilqr(dynamics: AffineDynamics, cost: QuadraticCostFunction, Vf: QuadraticValueFunction):
    """Backwards pass of the ILQR algorithm"""
    N = len(cost.c)
    scan_fun = lambda V, k: riccatiStep_ilqr(dynamics[k], cost[k], V)
    _, policy = jax.lax.scan(scan_fun, Vf, xs=jnp.arange(N), reverse=True)
    return policy


def riccatiStep_ddp(dynamics: QuadraticDynamics, cost: QuadraticCostFunction,
                    value: QuadraticValueFunction) -> tuple[QuadraticValueFunction, AffinePolicy]:
    """Perform one step of the backwards Ricatti recursion"""
    c, c_x, c_u, c_xx, c_ux, c_uu = cost
    v, v_x, v_xx = value

    _, f_x, f_u, _, _, _ = dynamics
    vf_xx, vf_ux, vf_uu = conditionQuadraticDynamics(dynamics, v_x)

    Q = c + v
    Q_x = c_x + f_x.T @ v_x
    Q_u = c_u + f_u.T @ v_x
    Q_xx = c_xx + f_x.T @ v_xx @ f_x + vf_xx
    Q_uu = c_uu + f_u.T @ v_xx @ f_u + vf_uu
    Q_ux = c_ux + f_u.T @ v_xx @ f_x + vf_ux

    l = -jnp.linalg.solve(Q_uu, Q_u)
    L = -jnp.linalg.solve(Q_uu, Q_ux)

    valueOut = QuadraticValueFunction(Q - 0.5 * l.T @ Q_uu @ l, Q_x - L.T @ Q_uu @ l, Q_xx - L.T @ Q_uu @ L)

    policy = AffinePolicy(l, L)
    return valueOut, policy


def backwardPass_ddp(dynamics: QuadraticDynamics, cost: QuadraticCostFunction, Vf: QuadraticValueFunction):
    """Backwards pass of the ILQR algorithm"""
    N = len(cost.c)
    scan_fun = lambda V, k: riccatiStep_ddp(dynamics[k], cost[k], V)
    _, policy = jax.lax.scan(scan_fun, Vf, xs=jnp.arange(N), reverse=True)
    return policy


def ensurePositiveDefinite(a, eps=1e-3):
    w, v = jnp.linalg.eigh(a)
    return (v * jnp.maximum(w, eps)) @ v.T


def conditionQuadraticCost(quadratic_cost: QuadraticCostFunction):
    """Ensure quadratic cost is strictly positive definite"""

    (c, c_x, c_u, c_xx, c_ux, c_uu) = quadratic_cost

    n = c_xx.shape[1]
    m = c_uu.shape[1]

    c_zz = jnp.block([[c_xx, c_ux.transpose(0, 2, 1)], [c_ux, c_uu]])
    c_zz = jax.vmap(ensurePositiveDefinite)(c_zz)
    c_xx, c_uu, c_ux = c_zz[:, :n, :n], c_zz[:, -m:, -m:], c_zz[:, -m:, :n]

    return QuadraticCostFunction(c, c_x, c_u, c_xx, c_ux, c_uu)


def conditionQuadraticDynamics(quadratic_dynamics: QuadraticDynamics, v_x: jnp.ndarray):
    """Ensure the quadratic terms of the DDP riccati step are positive definite"""
    _, _, _, f_xx, f_ux, f_uu = quadratic_dynamics
    vf_xx = jnp.einsum('i,ijk', v_x, f_xx)
    vf_uu = jnp.einsum('i,ijk', v_x, f_uu)
    vf_ux = jnp.einsum('i,ijk', v_x, f_ux)

    n = vf_xx.shape[0]
    m = vf_uu.shape[0]

    vf_zz = jnp.block([[vf_xx, vf_ux.T], [vf_ux, vf_uu]])
    vf_zz = ensurePositiveDefinite(vf_zz)
    vf_xx, vf_uu, vf_ux = vf_zz[:n, :n], vf_zz[-m:, -m:], vf_zz[-m:, :n]

    return vf_xx, vf_ux, vf_uu


def conditionValueFunction(Vf: QuadraticValueFunction):
    v, v_x, v_xx = Vf
    v_xx = ensurePositiveDefinite(Vf.v_xx)
    return QuadraticValueFunction(v, v_x, v_xx)


@partial(jax.jit, static_argnames=["dynamics", "runningCost", "terminalCost"])
def iterativeLqr(
    dynamics: Callable[[jnp.array, jnp.array], jnp.array],
    runningCost: Callable[[jnp.array, jnp.array], float],
    terminalCost: Callable[[jnp.array], float],
    x0: jnp.array,
    uGuess: jnp.array,
    maxIter=100,
    tol=1e-3
) -> tuple[Trajectory, jnp.ndarray, float, bool]:
    """
    Iterative LQR algorithm

    Arguments
    ---------
        dynamics : Discrete dynamics function: `xOut = f(x,u)`
        runningCost : Running cost function: `j = c(x,u)`
        terminalCost : Terminal cost function `j = cf(xf)`
        x0 : Initial state
        uGuess : Initial guess for control trajectory
        maxIter : Maximum number of ilqr iterations
        tol : Convergence tolerance: `abs(J_prev - J) <= tol`

    Returns
    -------
    traj : Output trajectory
    L : LQR feedback gains: `u[k] = L[k] @ (x[k]-xTraj[k]) + uTraj[k]`
    J : Corresponding cost
    converged : Whether ilqr converged
    """
    n = x0.shape[0]
    N, m = uGuess.shape
    cost = CostFunction(runningCost, terminalCost)
    policy = AffinePolicy(uGuess, jnp.zeros((N, m, n)))
    traj_prev = Trajectory(jnp.zeros((N + 1, n)), jnp.zeros((N, m)))

    # Rollout initial trajectory
    traj = trajectoryRollout(x0, dynamics, policy, traj_prev)
    J = cost(traj)

    # ILQR loop
    def ilqrCond(loopVars):
        traj, policy, J, converged, iter = loopVars
        return jnp.logical_not(converged) & (iter < maxIter)

    def ilqrStep(loopVars):
        traj, policy, J, converged, iter = loopVars

        affine_dynamics = AffineDynamics.from_trajectory(dynamics, traj)
        quadratic_cost = QuadraticCostFunction.from_trajectory(cost, traj)
        Vf = QuadraticValueFunction.fromTerminalCostFunction(cost, traj.xTraj[-1])

        quadratic_cost = conditionQuadraticCost(quadratic_cost)
        Vf = conditionValueFunction(Vf)

        policy = backwardPass_ilqr(affine_dynamics, quadratic_cost, Vf)
        traj_new, J_new = forwardPass2(x0, dynamics, cost, policy, traj)

        converged = abs(J - J_new) <= tol
        traj = traj_new
        J = J_new
        iter += 1
        return (traj, policy, J, converged, iter)

    out = jax.lax.while_loop(ilqrCond, ilqrStep, (traj, policy, J, False, 0))
    traj, policy, J, converged, iter = out

    return traj, policy.L, J, converged


@partial(jax.jit, static_argnames=["dynamics", "runningCost", "terminalCost"])
def differentialDynamicProgramming(
    dynamics: Callable[[jnp.array, jnp.array], jnp.array],
    runningCost: Callable[[jnp.array, jnp.array], float],
    terminalCost: Callable[[jnp.array], float],
    x0: jnp.array,
    uGuess: jnp.array,
    maxIter=100,
    tol=1e-3
) -> tuple[Trajectory, jnp.ndarray, float, bool]:
    """
    Differential dynamic programming algorithm

    Arguments
    ---------
        dynamics : Discrete dynamics function: `xOut = f(x,u)`
        runningCost : Running cost function: `j = c(x,u)`
        terminalCost : Terminal cost function `j = cf(xf)`
        x0 : Initial state
        uGuess : Initial guess for control trajectory
        maxIter : Maximum number of ilqr iterations
        tol : Convergence tolerance: `abs(J_prev - J) <= tol`

    Returns
    -------
    traj : Output trajectory
    L : LQR feedback gains: `u[k] = L[k] @ (x[k]-xTraj[k]) + uTraj[k]`
    J : Cost
    converged : Whether ddp converged
    """
    n = x0.shape[0]
    N, m = uGuess.shape
    cost = CostFunction(runningCost, terminalCost)
    policy = AffinePolicy(uGuess, jnp.zeros((N, m, n)))
    traj_prev = Trajectory(jnp.zeros((N + 1, n)), jnp.zeros((N, m)))

    # Rollout initial trajectory
    traj = trajectoryRollout(x0, dynamics, policy, traj_prev)
    J = cost(traj)

    # DDP loop
    def ddpCond(loopVars):
        traj, policy, J, converged, iter = loopVars
        return jnp.logical_not(converged) & (iter < maxIter)

    def ddpStep(loopVars):
        traj, policy, J, converged, iter = loopVars

        quadratic_dynamics = QuadraticDynamics.from_trajectory(dynamics, traj)
        quadratic_cost = QuadraticCostFunction.from_trajectory(cost, traj)
        Vf = QuadraticValueFunction.fromTerminalCostFunction(cost, traj.xTraj[-1])

        quadratic_cost = conditionQuadraticCost(quadratic_cost)
        Vf = conditionValueFunction(Vf)

        policy = backwardPass_ddp(quadratic_dynamics, quadratic_cost, Vf)
        traj_new, J_new = forwardPass2(x0, dynamics, cost, policy, traj)

        converged = abs(J - J_new) <= tol
        traj = traj_new
        J = J_new
        iter += 1
        return (traj, policy, J, converged, iter)

    out = jax.lax.while_loop(ddpCond, ddpStep, (traj, policy, J, False, 0))
    traj, policy, J, converged, iter = out

    return traj, policy.L, J, converged
