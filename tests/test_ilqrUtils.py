import jax
import jax.numpy as jnp
import pytest
import zProj.ilqrUtils as ilqr
import zProj.pytrees as pytrees

jax.config.update("jax_enable_x64", True)  # TEMP: Remove once iLQR line search improved


def test_trajectoryRollout():
    N = 3
    dynFun = lambda x, u: x + u
    policy = lambda x, k, alpha: jnp.array([alpha * k])
    xPrev = jnp.zeros(N)
    uPrev = jnp.zeros(N)
    trajPrev = (xPrev, uPrev)
    x0 = jnp.array([0.])

    xTraj, uTraj = ilqr.trajectoryRollout(x0, dynFun, policy, trajPrev)
    assert jnp.all(xTraj == jnp.array([0, 0, 1, 3])[:, None])
    assert jnp.all(uTraj == jnp.array([0, 1, 2])[:, None])

    xTraj, uTraj = ilqr.trajectoryRollout(x0, dynFun, policy, trajPrev, alpha=0.5)
    assert jnp.all(xTraj == jnp.array([0, 0, 0.5, 1.5])[:, None])
    assert jnp.all(uTraj == jnp.array([0, 0.5, 1])[:, None])


def test_forwardPass():
    """test that forward pass runs, no actual functional test"""
    x0 = jnp.array([1., 1])
    N = 3
    A = jnp.array([[1, 0], [1, 1]])
    B = jnp.array([[0], [1]])
    dynFun = lambda x, u: A @ x + B @ u
    costFun = lambda traj: jnp.sum(traj.xTraj**2) + jnp.sum(traj.uTraj**2)
    policy = lambda x, k, alpha: jnp.array([-alpha])
    trajPrev = ilqr.Trajectory(jnp.repeat(x0[None, :], N + 1, axis=0), jnp.zeros((N, 1)))
    dJFun = lambda alpha: 1
    JPrev = 1
    traj, JNew = ilqr.forwardPass(x0, dynFun, costFun, policy, trajPrev, dJFun, JPrev)
    assert isinstance(traj, pytrees.Trajectory)


def test_forwardPass2():
    x0 = jnp.array([1., 1])
    N = 3
    A = jnp.array([[1, 0], [1, 1]])
    B = jnp.array([[0], [1]])
    dynFun = lambda x, u: A @ x + B @ u
    costFun = lambda traj: jnp.sum(traj.xTraj**2) + jnp.sum(traj.uTraj**2)
    policy = lambda x, k, alpha: jnp.array([-10 * alpha])
    trajPrev = ilqr.Trajectory(jnp.repeat(x0[None, :], N + 1, axis=0), jnp.zeros((N, 1)))
    traj, J = ilqr.forwardPass2(x0, dynFun, costFun, policy, trajPrev)

    # Dummy checks, just verify we run without error
    assert isinstance(traj, pytrees.Trajectory)


def test_riccatiStep_ilqr():
    A = jnp.eye(2)
    B = jnp.eye(2)
    f = jnp.zeros(2)

    c = 0
    c_x = jnp.zeros(2)
    c_u = jnp.zeros(2)
    c_xx = jnp.eye(2)
    c_uu = jnp.eye(2)
    c_ux = jnp.zeros((2, 2))

    v = 0
    v_x = jnp.zeros(2)
    v_xx = c_xx

    dynamics = (f, A, B)
    cost = (c, c_x, c_u, c_xx, c_ux, c_uu)
    value = (v, v_x, v_xx)
    valueOut, policy = ilqr.riccatiStep_ilqr(dynamics, cost, value)

    assert valueOut.v == 0
    assert jnp.all(valueOut.v_x == jnp.array([0, 0]))
    assert jnp.all(valueOut.v_xx == 1.5 * jnp.eye(2))
    assert jnp.all(policy.l == jnp.array([0, 0]))
    assert jnp.all(policy.L == -0.5 * jnp.eye(2))


def test_backwardPass_ilqr():
    N = 2
    A = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    B = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    f = jnp.zeros((N, 2))

    c = jnp.zeros(N)
    c_x = jnp.zeros((N, 2))
    c_u = jnp.zeros((N, 2))
    c_xx = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    c_uu = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    c_ux = jnp.zeros((N, 2, 2))

    v = 0
    v_x = jnp.zeros(2)
    v_xx = c_xx[-1]

    dynamics = pytrees.AffineDynamics(f, A, B)
    cost = pytrees.QuadraticCostFunction(c, c_x, c_u, c_xx, c_ux, c_uu)
    value = pytrees.QuadraticValueFunction(v, v_x, v_xx)
    policy = ilqr.backwardPass_ilqr(dynamics, cost, value)

    # Dummy checks, just verify we run without error
    assert isinstance(policy, pytrees.AffinePolicy)

def test_riccatiStep_ddp():
    A = jnp.eye(2)
    B = jnp.eye(2)
    f = jnp.zeros(2)

    c = 0
    c_x = jnp.zeros(2)
    c_u = jnp.zeros(2)
    c_xx = jnp.eye(2)
    c_uu = jnp.eye(2)
    c_ux = jnp.zeros((2, 2))

    v = 0
    v_x = jnp.zeros(2)
    v_xx = c_xx

    dynamics = (f, A, B, jnp.zeros((2,2,2)), jnp.zeros((2,2,2)), jnp.zeros((2,2,2)))
    cost = (c, c_x, c_u, c_xx, c_ux, c_uu)
    value = (v, v_x, v_xx)
    valueOut, policy = ilqr.riccatiStep_ddp(dynamics, cost, value)

    assert valueOut.v == 0
    assert valueOut.v_x == pytest.approx(jnp.array([0, 0]))
    assert valueOut.v_xx == pytest.approx(1.5 * jnp.eye(2))
    assert policy.l == pytest.approx(jnp.array([0, 0]))
    assert policy.L == pytest.approx(-0.5 * jnp.eye(2), rel=1e-3)


def test_backwardPass_ddp():
    N = 2
    A = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    B = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    C = jnp.repeat(jnp.array([jnp.eye(2), jnp.eye(2)])[None, :, :, :], N, axis=0)
    D = jnp.repeat(jnp.array([jnp.eye(2), jnp.eye(2)])[None, :, :, :], N, axis=0)
    E = jnp.zeros((N,2,2,2))
    f = jnp.zeros((N, 2))

    c = jnp.zeros(N)
    c_x = jnp.zeros((N, 2))
    c_u = jnp.zeros((N, 2))
    c_xx = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    c_uu = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    c_ux = jnp.zeros((N, 2, 2))

    v = 0
    v_x = jnp.zeros(2)
    v_xx = c_xx[-1]

    dynamics = pytrees.QuadraticDynamics(f, A, B, C, D, E)
    cost = pytrees.QuadraticCostFunction(c, c_x, c_u, c_xx, c_ux, c_uu)
    value = pytrees.QuadraticValueFunction(v, v_x, v_xx)
    policy = ilqr.backwardPass_ddp(dynamics, cost, value)

    # Dummy checks, just verify we run without error
    assert isinstance(policy, pytrees.AffinePolicy)


def test_iterativeLqr():
    A = jnp.eye(2)
    B = jnp.eye(2)
    Q = jnp.eye(2)
    R = jnp.eye(2)
    N = 3
    dynamics = lambda x, u: A @ x + B @ u
    runningCost = lambda x, u: x @ Q @ x + u @ R @ u
    terminalCost = lambda x: x @ Q @ x
    x0 = jnp.array([2., 1])
    uGuess = jnp.zeros((N, 2))

    trajectory, J, converged = ilqr.iterativeLqr(dynamics, runningCost, terminalCost, x0, uGuess)
    assert converged

def test_differentialDynamicProgramming():
    A = jnp.eye(2)
    B = jnp.eye(2)
    Q = jnp.eye(2)
    R = jnp.eye(2)
    N = 3
    dynamics = lambda x, u: A @ x + B @ u
    runningCost = lambda x, u: x @ Q @ x + u @ R @ u
    terminalCost = lambda x: x @ Q @ x
    x0 = jnp.array([2., 1])
    uGuess = jnp.zeros((N, 2))

    trajectory, J, converged = ilqr.differentialDynamicProgramming(dynamics, runningCost, terminalCost, x0, uGuess)
    assert converged

test_backwardPass_ddp()
