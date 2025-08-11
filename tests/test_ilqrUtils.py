import numpy as np
import pytest
import zProj.ilqrUtils as ilqr


def dynFun(k, x, u):
    return x + u


def costFun(k, x, u):
    return np.sum(x**2) + np.sum(u**2)


def terminalCost(x):
    return np.sum(x**2)


def test_defaultInit():
    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))

    prob = ilqr.iLQR(dynFun, costFun, x0, u)
    assert prob.cf(x0) == 5  # check that terminal cost is computed correctly


def test_solve():
    """Check that solve runs without error"""

    x0 = np.array([1, 2])
    N = 3
    u = np.zeros((N, 2))
    prob = ilqr.iLQR(dynFun, costFun, x0, u, terminalCostFun=terminalCost)
    x, u, Pi = prob.solve()

    assert Pi(0, np.array([1, 1])).shape == (2,)
    with pytest.raises(IndexError):
        Pi(N, np.array([1, 1]))
