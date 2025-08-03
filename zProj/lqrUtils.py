import numpy as np
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
    R_inv: Callable[[float], np.ndarray],
    n: int
) -> np.ndarray:
    """LQR Hamilton Jacobia Bellman equation"""
    V = V.reshape((n, n))
    dV = -Q(t) + V @ B(t) @ R_inv(t) @ B(t).T @ V - V @ A(t) - A(t).T @ V
    dV = dV.reshape(-1)
    return dV


def finiteHorizonLqr(
    A: Callable[[float], np.ndarray],
    B: Callable[[float], np.ndarray],
    Q: Callable[[float], np.ndarray],
    R_inv: Callable[[float], np.ndarray],
    Qf: np.ndarray,
    T: float
) -> Callable[[float], np.ndarray]:
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
        R_inv : Inverse of control cost matrix as a function of time: `R_inv(t)`
        Qf : Terminal state cost matrix
        T : Time horizon

    Returns
    -------
        K : Optimal LQR gains as a function of time: `K(t)`
    """
    V0 = Qf.reshape(-1)
    n = A(0).shape[0]
    dV = lambda t, V: _lqrHjb(t, V, A, B, Q, R_inv, n)
    out = spi.solve_ivp(dV, (T, 0), V0, dense_output=True)
    K = lambda t: R_inv(t) @ B(t).T @ out.sol(t).reshape((n, n))
    return K


def infiniteHorizonIntegralLqr(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, Qi: np.ndarray, Ci: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute infinite horizon integral LQR gains for the standard decoupled state-control LQR cost function

    ```
    J = int_t( z^T Q_i z + x^T Q x + u^T R u)
    z = int_t( C_i x )
    xDot = A x + B u
    uLqr = -K x
    ```

    Arguments
    ---------
        A : State-space state matrix
        B : State-space input matrix
        Q : State cost matrix
        R : Control cost matrix
        Qi : Integral State cost matrix
        Ci: Matrix specifying integral states: `z = int_t( C_i x )`

    Returns
    -------
        Ki : Integral gains
        Kp : Proportional gains
    """
    n_i = Qi.shape[0]
    n_x, n_u = B.shape

    # Form integral augmented system
    Aw = np.block([[np.zeros((n_i, n_i)), Ci], [np.zeros((n_x, n_i)), A]])
    Bw = np.vstack([np.zeros((n_i, n_u)), B])
    Qw = spl.block_diag(Qi, Q)

    # Solve for LQR gains
    K = infiniteHorizonLqr(Aw, Bw, Qw, R)
    Ki = K[:, :n_i]
    Kp = K[:, n_i:]

    return Ki, Kp


## Basic LQR Controllers
def proportionalFeedbackController(x, x0, u0, K):
    control = -K @ (x - x0) + u0
    dxCtrl = np.array([])  # no controller states
    return control, dxCtrl
