import jax.numpy as jnp
import pytest
import zProj.pytrees as pytrees


def test_Trajectory():
    m = 2
    n = 3
    N = 4
    xTraj = jnp.arange((N + 1) * n).reshape((N + 1, n))
    uTraj = jnp.arange(N * m).reshape((N, m))
    traj = pytrees.Trajectory(xTraj, uTraj)
    assert jnp.all(traj.xTraj == xTraj)
    assert jnp.all(traj.uTraj == uTraj)

    for i in range(N):
        assert jnp.all(traj[i].xTraj == xTraj[i])
        assert jnp.all(traj[i].uTraj == uTraj[i])
    assert jnp.all(traj[i + 1].xTraj == xTraj[N])


def test_CostFunction():
    runningCost = lambda x, u: x @ x + u @ u
    terminalCost = lambda x: 2 * x @ x
    CostFun = pytrees.CostFunction(runningCost, terminalCost)

    traj = pytrees.Trajectory(jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 1]]))
    j0 = CostFun(traj, k=0)
    J = CostFun(traj)
    assert j0 == 7
    assert J == 57


def test_CostFunction_runningOnly():
    runningCost = lambda x, u: x @ x + u @ u
    CostFun = pytrees.CostFunction.runningOnly(runningCost, 2)

    traj = pytrees.Trajectory(jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 1]]))
    j0 = CostFun(traj, k=0)
    J = CostFun(traj)
    assert j0 == 7
    assert J == 32


def test_QuadraticValueFunction():
    v = jnp.array(1.)
    v_x = jnp.array([2., 3])
    v_xx = jnp.array([[4., 5], [6, 7]])
    V = pytrees.QuadraticValueFunction(v, v_x, v_xx)
    assert V.v == v
    assert jnp.all(V.v_x == v_x)
    assert jnp.all(V.v_xx == v_xx)

    x = jnp.array([8., 9])
    V_exp = 1 + 16 + 27 + 807.5
    assert V(x) == pytest.approx(V_exp)


def test_QuadraticValueFunction_fromTerminalCostFunction():

    c = 1
    c_x = jnp.array([1, 2])
    c_xx = jnp.eye(2)
    x0 = jnp.zeros(2)
    costFun = pytrees.CostFunction(0, lambda x: c + c_x.T @ x + 0.5 * x.T @ c_xx @ x)
    value = pytrees.QuadraticValueFunction.fromTerminalCostFunction(costFun, x0)
    assert value.v == c
    assert jnp.all(value.v_x == c_x)
    assert jnp.all(value.v_xx == c_xx)


def test_QuadraticCostFunction_base():
    c = jnp.array(0.)
    c_x = jnp.array([1, 2])
    c_u = jnp.array([2, 1])
    c_xx = jnp.eye(2)
    c_ux = jnp.ones((2, 2))
    c_uu = jnp.eye(2)
    C = pytrees.QuadraticCostFunction(c, c_x, c_u, c_xx, c_ux, c_uu)
    assert C.c == c
    assert jnp.all(C.c_x == c_x)
    assert jnp.all(C.c_u == c_u)
    assert jnp.all(C.c_xx == c_xx)
    assert jnp.all(C.c_ux == c_ux)
    assert jnp.all(C.c_uu == c_uu)

    x = jnp.array([1., 2])
    u = jnp.array([3., 4])
    C_exp = 0. + 5 + 10 + 2.5 + 21 + 12.5
    assert C(x, u) == pytest.approx(C_exp)


def test_QuadraticCostFunction_getItem():
    c = jnp.array([1, 1])
    c_x = jnp.eye(2)
    c_u = jnp.eye(2)
    c_xx = jnp.array([jnp.eye(2), jnp.zeros((2, 2))])
    c_ux = jnp.array([jnp.eye(2), jnp.zeros((2, 2))])
    c_uu = jnp.array([jnp.eye(2), jnp.zeros((2, 2))])
    C = pytrees.QuadraticCostFunction(c, c_x, c_u, c_xx, c_ux, c_uu)

    C0 = C[0]
    assert C0.c == c[0]
    assert jnp.all(C0.c_x == c_x[0])
    assert jnp.all(C0.c_u == c_u[0])
    assert jnp.all(C0.c_xx == c_xx[0])
    assert jnp.all(C0.c_ux == c_ux[0])
    assert jnp.all(C0.c_uu == c_uu[0])

    C1 = C[1]
    assert C1.c == c[1]
    assert jnp.all(C1.c_x == c_x[1])
    assert jnp.all(C1.c_u == c_u[1])
    assert jnp.all(C1.c_xx == c_xx[1])
    assert jnp.all(C1.c_ux == c_ux[1])
    assert jnp.all(C1.c_uu == c_uu[1])


def test_QuadraticCostFunction_fromFunction():
    c = 1
    c_x = jnp.array([1, 2])
    c_u = jnp.array([2, 1])
    c_xx = jnp.eye(2)
    c_ux = jnp.array([[1, 2], [3, 4]])
    c_uu = jnp.eye(2)
    x0 = jnp.zeros(2)
    u0 = jnp.zeros(2)

    costFun = pytrees.CostFunction.runningOnly(
        lambda x, u: c + c_x.T @ x + c_u.T @ u + 0.5 * (x.T @ c_xx @ x + 2 * u.T @ c_ux @ x + u.T @ c_uu @ u)
    )
    C = pytrees.QuadraticCostFunction.from_function(costFun, x0, u0)

    assert C.c == c
    assert jnp.all(C.c_x == c_x)
    assert jnp.all(C.c_u == c_u)
    assert jnp.all(C.c_xx == c_xx)
    assert jnp.all(C.c_ux == c_ux)
    assert jnp.all(C.c_uu == c_uu)


def test_QuadraticCostFunction_fromTrajectory():
    c = 1
    c_x = jnp.array([1, 2])
    c_u = jnp.array([2, 1])
    c_xx = jnp.eye(2)
    c_ux = jnp.array([[1, 2], [3, 4]])
    c_uu = jnp.eye(2)
    x0 = jnp.array([[0., 0], [1, 0], [1, 1]])
    u0 = jnp.zeros((2, 2))
    traj = pytrees.Trajectory(x0, u0)
    costFun = pytrees.CostFunction.runningOnly(
        lambda x, u: c + c_x.T @ x + c_u.T @ u + 0.5 * (x.T @ c_xx @ x + 2 * u.T @ c_ux @ x + u.T @ c_uu @ u)
    )
    C = pytrees.QuadraticCostFunction.from_trajectory(costFun, traj)

    C0 = C[0]
    assert C0.c == c
    assert jnp.all(C0.c_x == c_x)
    assert jnp.all(C0.c_u == c_u)
    assert jnp.all(C0.c_xx == c_xx)
    assert jnp.all(C0.c_ux == c_ux)
    assert jnp.all(C0.c_uu == c_uu)

    C1 = C[1]
    x1 = x0[1]
    assert C1.c == costFun(traj, k=1)
    assert jnp.all(C1.c_x == c_x + c_xx @ x1)
    assert jnp.all(C1.c_u == c_u + c_ux @ x1)
    assert jnp.all(C1.c_xx == c_xx)
    assert jnp.all(C1.c_ux == c_ux)
    assert jnp.all(C1.c_uu == c_uu)


def test_AffineDynamics_base():
    f = jnp.array([1, 1])
    f_x = jnp.array([[2, 3], [4, 5]])
    f_u = jnp.array([[6], [7]])
    dynamics = pytrees.AffineDynamics(f, f_x, f_u)
    assert jnp.all(dynamics.f == f)
    assert jnp.all(dynamics.f_x == f_x)
    assert jnp.all(dynamics.f_u == f_u)

    x = jnp.array([1, 2])
    u = jnp.array([2])
    xOut_exp = jnp.array([21, 29])
    assert dynamics(x, u) == pytest.approx(xOut_exp)


def test_AffineDynamics_getItem():
    f = jnp.array([0, 1])
    f_x = jnp.eye(2)
    f_u = jnp.eye(2)
    dyn = pytrees.AffineDynamics(f, f_x, f_u)

    dyn0 = dyn[0]
    assert dyn0.f == f[0]
    assert jnp.all(dyn0.f_x == f_x[0])
    assert jnp.all(dyn0.f_u == f_u[0])

    dyn1 = dyn[1]
    assert dyn1.f == f[1]
    assert jnp.all(dyn1.f_x == f_x[1])
    assert jnp.all(dyn1.f_u == f_u[1])


def test_AffineDynamics_fromFunction():
    f = jnp.array([1, 1])
    f_x = jnp.array([[2, 3], [4, 5]])
    f_u = jnp.array([[6], [7]])
    dynFun = lambda x, u: f + f_x @ x + f_u @ u
    x0 = jnp.zeros(2)
    u0 = jnp.zeros(1)

    dynamics = pytrees.AffineDynamics.from_function(dynFun, x0, u0)
    assert jnp.all(dynamics.f == f)
    assert jnp.all(dynamics.f_x == f_x)
    assert jnp.all(dynamics.f_u == f_u)


def test_AffineDynamics_fromTrajectory():
    f = jnp.array([1, 1])
    f_x = jnp.array([[2, 3], [4, 5]])
    f_u = jnp.array([[6], [7]])
    dynFun = lambda x, u: f + f_x @ x + f_u @ u + 0.5 * x.T @ x
    x0 = jnp.array([[0., 0], [1, 0], [2, 0]])
    u0 = jnp.zeros((2, 1))
    traj = pytrees.Trajectory(x0, u0)

    dynamics = pytrees.AffineDynamics.from_trajectory(dynFun, traj)
    assert jnp.all(dynamics[0].f == f)
    assert jnp.all(dynamics[0].f_x == f_x)
    assert jnp.all(dynamics[0].f_u == f_u)
    assert jnp.all(dynamics[1].f == dynFun(x0[1], u0[1]))
    assert jnp.all(dynamics[1].f_x == f_x + x0[1])
    assert jnp.all(dynamics[1].f_u == f_u)


def test_AffinePolicy_base():
    l = jnp.array([1, 2])
    L = jnp.array([[1, 2], [3, 4]])
    policy = pytrees.AffinePolicy(l, L)

    x = jnp.array([1, 2])
    u_exp = jnp.array([6, 13])
    assert jnp.all(policy(x) == u_exp)


def test_AffinePolicy_multi():
    n = 2
    m = 3
    N = 2
    l = jnp.arange(0, N * m).reshape((N, m))
    L = jnp.arange(0, N * m * n).reshape((N, m, n))
    x = jnp.array([0, 0])
    policy = pytrees.AffinePolicy(l, L)

    assert jnp.all(policy[1].l == l[1])
    assert jnp.all(policy[0].L == L[0])
    assert jnp.all(policy(x, k=0) == l[0])
    assert jnp.all(policy(x, k=1) == l[1])
    assert jnp.all(policy(x, k=0, alpha=0.5) == 0.5 * l[0])


def test_QuadraticDeltaCost():
    dJ_lin = 1
    dJ_quad = 2
    dJFun = pytrees.QuadraticDeltaCost(dJ_lin, dJ_quad)
    assert dJFun(1) == 3
    assert dJFun(0.5) == 1
