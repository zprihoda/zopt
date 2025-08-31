import jax.numpy as jnp
import numpy as np
import pytest
import zProj.ilqrUtils as ilqr

import jax

jax.config.update("jax_enable_x64", True)  # TEMP: Remove once iLQR line search improved


def dynFun(k, x, u):
    return x + u


def costFun(k, x, u):
    return np.sum(x**2) + np.sum(u**2)


def terminalCost(x):
    return np.sum(x**2)


def test_QuadraticValueFunction():
    v = jnp.array(1.)
    v_x = jnp.array([2., 3])
    v_xx = jnp.array([[4., 5], [6, 7]])
    V = ilqr.QuadraticValueFunction(v, v_x, v_xx)
    assert V.v == v
    assert jnp.all(V.v_x == v_x)
    assert jnp.all(V.v_xx == v_xx)

    x = jnp.array([8., 9])
    V_exp = 1 + 16 + 27 + 807.5
    assert V(x) == pytest.approx(V_exp)


def test_QuadraticCostFunction_base():
    c = jnp.array(0.)
    c_x = jnp.array([1, 2])
    c_u = jnp.array([2, 1])
    c_xx = jnp.eye(2)
    c_xu = jnp.ones((2, 2))
    c_uu = jnp.eye(2)
    C = ilqr.QuadraticCostFunction(c, c_x, c_u, c_xx, c_xu, c_uu)
    assert C.c == c
    assert jnp.all(C.c_x == c_x)
    assert jnp.all(C.c_u == c_u)
    assert jnp.all(C.c_xx == c_xx)
    assert jnp.all(C.c_xu == c_xu)
    assert jnp.all(C.c_uu == c_uu)

    x = jnp.array([1., 2])
    u = jnp.array([3., 4])
    C_exp = 0. + 5 + 10 + 2.5 + 21 + 12.5
    assert C(x, u) == pytest.approx(C_exp)


def test_QuadraticCostFunction_fromFunction():
    c = 1
    c_x = jnp.array([1, 2])
    c_u = jnp.array([2, 1])
    c_xx = jnp.eye(2)
    c_xu = jnp.array([[1, 2], [3, 4]])
    c_uu = jnp.eye(2)
    x0 = jnp.zeros(2)
    u0 = jnp.zeros(2)
    costFun = lambda x, u: c + c_x.T @ x + c_u.T @ u + 0.5 * (x.T @ c_xx @ x + 2 * x.T @ c_xu @ u + u.T @ c_uu @ u)
    C = ilqr.QuadraticCostFunction.from_function(costFun, x0, u0)

    assert C.c == c
    assert jnp.all(C.c_x == c_x)
    assert jnp.all(C.c_u == c_u)
    assert jnp.all(C.c_xx == c_xx)
    assert jnp.all(C.c_xu == c_xu)
    assert jnp.all(C.c_uu == c_uu)


def test_AffineDynamics_base():
    f = jnp.array([1, 1])
    f_x = jnp.array([[2, 3], [4, 5]])
    f_u = jnp.array([[6], [7]])
    dynamics = ilqr.AffineDynamics(f, f_x, f_u)
    assert jnp.all(dynamics.f == f)
    assert jnp.all(dynamics.f_x == f_x)
    assert jnp.all(dynamics.f_u == f_u)

    x = jnp.array([1, 2])
    u = jnp.array([2])
    xOut_exp = jnp.array([21, 29])
    assert dynamics(x, u) == pytest.approx(xOut_exp)


def test_AffineDynamics_fromFunction():
    f = jnp.array([1, 1])
    f_x = jnp.array([[2, 3], [4, 5]])
    f_u = jnp.array([[6], [7]])
    dynFun = lambda x, u: f + f_x @ x + f_u @ u
    x0 = jnp.zeros(2)
    u0 = jnp.zeros(1)

    dynamics = ilqr.AffineDynamics.from_function(dynFun, x0, u0)
    assert jnp.all(dynamics.f == f)
    assert jnp.all(dynamics.f_x == f_x)
    assert jnp.all(dynamics.f_u == f_u)


def test_iLqrDefaultInit():
    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))

    prob = ilqr.iLQR(dynFun, costFun, x0, u)
    assert prob.cf(x0) == 5  # check that terminal cost is computed correctly


def test_iLqrSolve():
    """Check that solve runs without error"""

    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))
    prob = ilqr.iLQR(dynFun, costFun, x0, u, terminalCostFun=terminalCost)
    x, u, LArr = prob.solve()
    assert LArr.shape == (N, 2, 2)


def test_ddpDefaultInit():
    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))

    prob = ilqr.DDP(dynFun, costFun, x0, u)
    assert prob.cf(x0) == 5  # check that terminal cost is computed correctly


def test_ddpSolve():
    """Check that solve runs without error"""

    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))
    prob = ilqr.DDP(dynFun, costFun, x0, u, terminalCostFun=terminalCost)
    x, u, LArr = prob.solve()
    assert LArr.shape == (N, 2, 2)
