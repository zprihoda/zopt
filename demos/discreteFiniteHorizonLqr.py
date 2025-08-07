import matplotlib.pyplot as plt
import numpy as np

from zProj.quadcopter import Quadcopter
from zProj.simulator import Simulator, SimBlock
from zProj.plottingTools import plotTimeTrajectory
from zProj.lqrUtils import discreteFiniteHorizonLqr, proportionalFeedbackController


def main():
    # User inputs
    uvwTrim = np.zeros(3)
    Q = np.eye(8)
    R = np.eye(4)
    Qf = 10 * np.eye(8)
    x0 = np.zeros(12)
    x0[0:3] = 1
    T = 10
    dt = 0.1

    # Get linearized system
    ac = Quadcopter()
    xTrim, uTrim = ac.trim(uvwTrim)
    A, B = ac.linearize(xTrim, uTrim, dt=dt)

    # Design LQR controller
    N = int(T / dt)
    Ak = np.repeat(A[None, :, :], N, axis=0)
    Bk = np.repeat(B[None, :, :], N, axis=0)
    Qk = np.repeat(Q[None, :, :], N-1, axis=0)
    Qk = np.concatenate([Qf[None,:,:], Qk], axis=0)
    Rk = np.repeat(R[None, :, :], N, axis=0)
    K = discreteFiniteHorizonLqr(Ak, Bk, Qk, Rk, N)
    xCtrl0 = np.array([])

    # Simple Simulation
    dynamics = SimBlock(lambda k, x, u: (None, x + dt * ac.inertialDynamics(x, u)), x0, dt=dt, name="Dynamics")
    controller = SimBlock(
        lambda k, xCtrl, x: proportionalFeedbackController(x[:8], xTrim, uTrim, K[k]),
        xCtrl0,
        dt=dt,
        name="Controller",
        jittable=False
    )
    t_span = (0, T)
    sim = Simulator([controller, dynamics], t_span)
    tArr, _, xArr, uArr, _ = sim.simulate()

    # Plot Results
    plotTimeTrajectory(tArr, xArr[:, 0:3], names=['u', 'v', 'w'], title="Body Velocities")
    plotTimeTrajectory(tArr, xArr[:, 3:6], names=['p', 'q', 'r'], title="Body Rates")
    plotTimeTrajectory(tArr, xArr[:, 6:9], names=['phi', 'theta', 'psi'], title="Euler Angles")
    plotTimeTrajectory(tArr, xArr[:, 9:12], names=['x', 'y', 'z'], title="Positions")
    plotTimeTrajectory(tArr[:-1], uArr, names=["thrust", "pDot", "qDot", "rDot"], title="Pseudo Controls")
    plt.show()


if __name__ == "__main__":
    main()
