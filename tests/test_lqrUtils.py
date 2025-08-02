import numpy as np
import pytest

import zProj.lqrUtils as lqr


def test_computeInfiniteHorizonLqrGains():
    A = np.eye(2)
    B = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    K = lqr.computeInfiniteHorizonLqrGains(A, B, Q, R)
    K_exp = (1 + np.sqrt(2)) * np.eye(2)
    assert K == pytest.approx(K_exp)


def test_infiniteHorizonLqrController():
    K = np.array([1, 1])
    x0 = np.zeros(2)
    u0 = np.array([1])
    x = np.ones(2)
    u = lqr.infiniteHorizonLqrController(x, x0, u0, K)
    u_exp = np.array([-1])
    assert u == pytest.approx(u_exp)
