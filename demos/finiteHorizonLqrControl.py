import matplotlib.pyplot as plt
import numpy as np

from zProj.quadcopter import Quadcopter
from zProj.simulator import Simulator, SimBlock
from zProj.plottingTools import plotTimeTrajectory
from zProj.lqrUtils import finiteHorizonLqr, proportionalFeedbackController


def main():
    # User inputs
    uvwTrim = np.zeros(3)
    Q = np.eye(8)
    R = np.eye(4)
    Qf = 10 * np.eye(8)
    x0 = np.zeros(12)
    x0[0:3] = 1
    T = 5
    dt = 0.1

    # Get linearized system
    ac = Quadcopter()
    xTrim, uTrim = ac.trim(uvwTrim)
    A, B = ac.linearize(xTrim, uTrim)

    # Design LQR controller
    At = lambda t: A
    Bt = lambda t: B
    Qt = lambda t: Q
    Rt = lambda t: R
    K = finiteHorizonLqr(At, Bt, Qt, Rt, Qf, T)

    # Simple Simulation
    dynamics = SimBlock(lambda t, x, u: (None, ac.inertialDynamics(x, u)), x0, name="Dynamics")
    controller = SimBlock(
        lambda t, xCtrl, x: proportionalFeedbackController(x[:8], xTrim, uTrim, K(t)), np.array([]), name="Controller"
    )

    t_span = (0, T)
    t_eval = np.arange(0, T, dt)
    sim = Simulator([controller, dynamics], t_span, t_eval=t_eval)
    tArr, _, xArr, uArr, _ = sim.simulate()

    # Plot Results
    plotTimeTrajectory(tArr, xArr[:, 0:3], names=['u', 'v', 'w'], title="Body Velocities")
    plotTimeTrajectory(tArr, xArr[:, 3:6], names=['p', 'q', 'r'], title="Body Rates")
    plotTimeTrajectory(tArr, xArr[:, 6:9], names=['phi', 'theta', 'psi'], title="Euler Angles")
    plotTimeTrajectory(tArr, xArr[:, 9:12], names=['x', 'y', 'z'], title="Positions")
    plotTimeTrajectory(tArr, uArr, names=["thrust", "pDot", "qDot", "rDot"], title="Pseudo Controls")
    plt.show()


if __name__ == "__main__":
    main()
