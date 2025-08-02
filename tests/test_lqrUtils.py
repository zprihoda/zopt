import numpy as np
import pytest

import zProj.lqrUtils as lqr


def test_infiniteHorizonLqr():
    A = np.eye(2)
    B = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    K = lqr.infiniteHorizonLqr(A, B, Q, R)
    K_exp = (1 + np.sqrt(2)) * np.eye(2)
    assert K == pytest.approx(K_exp)


def test_lqrHjb():
    A = lambda t: np.eye(2)
    B = lambda t: np.eye(2)
    Q = lambda t: np.eye(2)
    R = lambda t: np.eye(2)
    V = np.eye(2)
    n = 2
    t = 0
    dV = lqr._lqrHjb(t, V, A, B, Q, R, n)
    dV_exp = -2 * np.eye(2).reshape(-1)
    assert dV == pytest.approx(dV_exp)


def test_finiteHorizonLqr():
    A = lambda t: np.eye(2)
    B = lambda t: np.eye(2)
    Q = lambda t: np.eye(2)
    R = lambda t: np.eye(2)
    Qf = np.eye(2)
    T = 1
    K = lqr.finiteHorizonLqr(A, B, Q, R, Qf, T)
    assert K(T) == pytest.approx(np.eye(2))  # From V(T) = Qf, K(T) = R(T)^{-1} @ B(T).T @ V(T)

    # Analytical solution to the scalar form of the above Ricatti equation
    K_exp = lambda t: ((1+np.sqrt(2))*np.exp(2*np.sqrt(2)) - (np.sqrt(2) - 1)*np.exp(2*np.sqrt(2)*t)) / \
        (np.exp(2*np.sqrt(2)*t) + np.exp(2*np.sqrt(2)))
    assert K(0) == pytest.approx(K_exp(0) * np.eye(2), rel=1e-3)


def test_proportionalFeedbackController():
    K = np.array([1, 1])
    x0 = np.zeros(2)
    u0 = np.array([1])
    x = np.ones(2)
    u = lqr.proportionalFeedbackController(x, x0, u0, K)
    u_exp = np.array([-1])
    assert u == pytest.approx(u_exp)
