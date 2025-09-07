import jax
import jax.numpy as jnp
import numpy as np
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
    assert isinstance(J, float)


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
    assert jnp.all(valueOut.v_xx == -2 * jnp.eye(2))
    assert jnp.all(policy.l == jnp.array([0, 0]))
    assert jnp.all(policy.L == -2 * jnp.eye(2))


def test_backwardPass():
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


### OLD Tests
def dynFun(k, x, u):
    return x + u


def costFun(k, x, u):
    return np.sum(x**2) + np.sum(u**2)


def terminalCost(x):
    return np.sum(x**2)


def old_test_iLqrDefaultInit():
    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))

    prob = ilqr.iLQR(dynFun, costFun, x0, u)
    assert prob.cf(x0) == 5  # check that terminal cost is computed correctly


def old_test_iLqrSolve():
    """Check that solve runs without error"""

    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))
    prob = ilqr.iLQR(dynFun, costFun, x0, u, terminalCostFun=terminalCost)
    x, u, LArr = prob.solve()
    assert LArr.shape == (N, 2, 2)


def old_test_ddpDefaultInit():
    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))

    prob = ilqr.DDP(dynFun, costFun, x0, u)
    assert prob.cf(x0) == 5  # check that terminal cost is computed correctly


def old_test_ddpSolve():
    """Check that solve runs without error"""

    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))
    prob = ilqr.DDP(dynFun, costFun, x0, u, terminalCostFun=terminalCost)
    x, u, LArr = prob.solve()
    assert LArr.shape == (N, 2, 2)
