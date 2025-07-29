import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi

from quadcopter import Quadcopter


def computeInfiniteHorizonLqrGains(A, B, Q, R):
    P = spl.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


@jax.jit
def lqrController(x, x0, u0, K):
    x_fb = x[:8]
    control = -K @ (x_fb - x0) + u0
    return control


def step(x: np.ndarray, controlFun, dynFun, dt):
    u = controlFun(x)
    out = spi.solve_ivp(dynFun, [0, dt], x, t_eval=[dt], args=(u,))
    xOut = out.y[:, 0]
    return xOut, u


def plotResults(tArr, xArr, uArr):
    stateGroupNames = ["Body Velocities", "Body Rates", "Euler Angles", "Positions"]
    stateNames = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']
    controlNames = ['thrust', 'pDot', 'qDot', 'rDot']

    Nx = 3
    for iFig in range(len(stateNames) // Nx):
        fig, axs = plt.subplots(Nx, 1, sharex=True)
        for jx in range(Nx):
            ix = iFig * Nx + jx
            axs[jx].plot(tArr, xArr[:-1, ix])
            axs[jx].set_ylabel(stateNames[ix])
            axs[jx].grid()
        axs[Nx - 1].set_xlabel("time")
        fig.suptitle(stateGroupNames[iFig])

    fig, axs = plt.subplots(4, 1, sharex=True)
    for iu in range(4):
        axs[iu].plot(tArr, uArr[:, iu])
        axs[iu].set_ylabel(controlNames[iu])
        axs[iu].grid()
    axs[3].set_xlabel("time")
    fig.suptitle("Control")


def main():
    # User inputs
    uvwTrim = np.zeros(3)
    Q = np.eye(8)
    R = np.eye(4)
    x0 = np.zeros(12)
    x0[0:3] = 1
    T = 10
    dt = 0.1

    # Get linearized system
    ac = Quadcopter()
    xTrim, uTrim = ac.trim(uvwTrim)
    A, B = ac.linearize(xTrim, uTrim)
    xTrim = xTrim[:8]
    A = A[:8, :8]
    B = B[:8, :]

    # Design LQR controller
    K = computeInfiniteHorizonLqrGains(A, B, Q, R)

    # Simple Simulation
    controlFun = jax.jit(lambda x: lqrController(x, xTrim, uTrim, K))
    dynFun = jax.jit(lambda t, x, u: ac.inertialDynamics(x, u))

    nx = 12
    nu = 4

    tArr = np.arange(0, T, dt)
    xArr = np.zeros((len(tArr) + 1, nx))
    uArr = np.zeros((len(tArr), nu))
    xArr[0] = x0

    for i in range(len(tArr)):
        xArr[i + 1], uArr[i] = step(xArr[i], controlFun, dynFun, dt)

    plotResults(tArr, xArr, uArr)
    plt.show()


if __name__ == "__main__":
    main()
