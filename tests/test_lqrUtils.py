import jax.numpy as jnp
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


def test_infiniteHorizonIntegralLqr():
    A = np.eye(2)
    B = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    Qi = np.eye(1)
    Ci = np.array([1, 0])
    Ki, Kp = lqr.infiniteHorizonIntegralLqr(A, B, Q, R, Qi, Ci)
    Ki_exp = np.array([[1], [0]])
    Kp_exp = np.diag([3, 1 + np.sqrt(2)])
    assert Ki == pytest.approx(Ki_exp)
    assert Kp == pytest.approx(Kp_exp)


def test_discreteFiniteHorizonLqr():
    N = 2
    A = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    B = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    Q = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    R = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    K = lqr.discreteFiniteHorizonLqr(A, B, Q, R, N)
    assert K[1] == pytest.approx(0.5 * np.eye(2))
    assert K[0] == pytest.approx(0.6 * np.eye(2))


def test_discreteInfiniteHorizonLqr():
    A = np.eye(2)
    B = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    K = lqr.discreteInfiniteHorizonLqr(A, B, Q, R)
    K_exp = (1 + np.sqrt(5)) / (3 + np.sqrt(5)) * np.eye(2)  # Analytical solution, v = (1 + sqrt(5))/2, K = v/(v+1)
    assert K == pytest.approx(K_exp)


def test_bilinearAffineLqr():
    N = 2
    A = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    B = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    d = jnp.repeat(jnp.ones(2)[None, :], N, axis=0)
    Q = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    R = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    H = jnp.repeat(jnp.eye(2)[None, :, :], N, axis=0)
    q = jnp.repeat(jnp.ones(2)[None, :], N, axis=0)
    r = jnp.repeat(jnp.ones(2)[None, :], N, axis=0)
    q0 = jnp.ones(N)
    K, k = lqr.bilinearAffineLqr(A, B, d, Q, R, H, q, r, q0, N)

    assert K[1] == pytest.approx(jnp.eye(2))
    assert k[1] == pytest.approx(1.5 * jnp.ones(2))
    assert K[0] == pytest.approx(jnp.eye(2))
    assert k[0] == pytest.approx(jnp.ones(2))


def test_proportionalFeedbackController():
    K = np.array([1, 1])
    x0 = np.zeros(2)
    u0 = np.array([1])
    x = np.ones(2)
    u, _ = lqr.proportionalFeedbackController(x, x0, u0, K)
    u_exp = np.array([-1])
    assert u == pytest.approx(u_exp)


test_bilinearAffineLqr()
