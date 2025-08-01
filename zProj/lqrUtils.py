import numpy as np
import scipy.linalg as spl


def computeInfiniteHorizonLqrGains(A, B, Q, R):
    P = spl.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


def infiniteHorizonLqrController(x, x0, u0, K):
    control = -K @ (x - x0) + u0
    return control
