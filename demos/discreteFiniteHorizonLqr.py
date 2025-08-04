import matplotlib.pyplot as plt
import numpy as np

from zProj.quadcopter import Quadcopter
from zProj.simulator import Simulator
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
    Qk = np.repeat(Q[None, :, :], N, axis=0)
    Rk = np.repeat(R[None, :, :], N, axis=0)
    K = discreteFiniteHorizonLqr(Ak, Bk, Qk, Rk, Qf, N)
    xCtrl0 = np.array([])

    # Simple Simulation
    # TODO: Add discrete functionality to simulator
    # - Create generic dynamics / controller classes
    # - Controller class has a dt field specifying continuous or discrete time
    dyn_fun = lambda t, x, u: ac.inertialDynamics(x, u)
    control_fun = lambda k, x, xCtrl: proportionalFeedbackController(x[:8], xTrim, uTrim, K[k])
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)
    sim = Simulator(dyn_fun, control_fun, t_span, x0, xCtrl0, t_eval=t_eval)
    tArr, xArr, _, uArr = sim.simulate()

    # Plot Results
    plotTimeTrajectory(tArr, xArr[:, 0:3], names=['u', 'v', 'w'], title="Body Velocities")
    plotTimeTrajectory(tArr, xArr[:, 3:6], names=['p', 'q', 'r'], title="Body Rates")
    plotTimeTrajectory(tArr, xArr[:, 6:9], names=['phi', 'theta', 'psi'], title="Euler Angles")
    plotTimeTrajectory(tArr, xArr[:, 9:12], names=['x', 'y', 'z'], title="Positions")
    plotTimeTrajectory(tArr, uArr, names=["thrust", "pDot", "qDot", "rDot"], title="Pseudo Controls")
    plt.show()


if __name__ == "__main__":
    main()
