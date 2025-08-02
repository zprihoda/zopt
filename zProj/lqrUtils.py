import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.integrate as spi

from typing import Callable


## LQR Algorithms
def infiniteHorizonLqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute infinite horizon LQR gains for the standard decoupled state-control LQR cost function

    ```
    J = int_t(x^T Q x + u^T R u)
    xDot = A x + B u
    uLqr = -K x
    ```

    Arguments
    ---------
        A : State-space state matrix
        B : State-space input matrix
        Q : State cost matrix
        R : Control cost matrix

    Returns
    -------
        K : LQR gains
    """
    P = spl.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


def _lqrHjb(
        t: float,
        V: np.ndarray,
        A: Callable[[float], np.ndarray],
        B: Callable[[float], np.ndarray],
        Q: Callable[[float], np.ndarray],
        R: Callable[[float], np.ndarray],
        n: int) -> np.ndarray:
    """LQR Hamilton Jacobia Bellman equation"""
    # TODO: Resolve inverse. Either: use npl.solve, pass R_inv function directly as argument
    V = V.reshape((n, n))
    dV = -Q(t) + V @ B(t) @ npl.inv(R(t)) @ B(t).T @ V - V @ A(t) - A(t).T @ V
    dV = dV.reshape(-1)
    return dV


def finiteHorizonLqr(
        A: Callable[[float], np.ndarray],
        B: Callable[[float], np.ndarray],
        Q: Callable[[float], np.ndarray],
        R: Callable[[float], np.ndarray],
        Qf: np.ndarray,
        T: float) -> Callable[[float], np.ndarray]:
    """
    Compute the finite horizon LQR gains by numerically integrating the LQR HJB equation

    ```
    J = xf^T Qf xf + int_t(x^T Q x + u^T R u)
    xDot = A x + B u
    uLqr = -K x
    ```

    Arguments
    ---------
        A : State-space state matrix as a function of time: `A(t)`
        B : State-space input matrix as a function of time: `B(t)`
        Q : State cost matrix as a function of time: `Q(t)`
        R : Control cost matrix as a function of time: `R(t)`
        Qf : Terminal state cost matrix
        T : Time horizon

    Returns
    -------
        K : Optimal LQR gains as a function of time: `K(t)`
    """
    V0 = Qf.reshape(-1)
    n = A(0).shape[0]
    dV = lambda t, V: _lqrHjb(t, V, A, B, Q, R, n)
    out = spi.solve_ivp(dV, (T, 0), V0, dense_output=True)
    K = lambda t: npl.inv(R(t)) @ B(t).T @ out.sol(t).reshape((n, n))
    return K


## Basic LQR Controllers
def proportionalFeedbackController(x, x0, u0, K):
    control = -K @ (x - x0) + u0
    return control
