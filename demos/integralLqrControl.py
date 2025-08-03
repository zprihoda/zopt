import matplotlib.pyplot as plt
import numpy as np

from zProj.quadcopter import Quadcopter
from zProj.simulator import Simulator
from zProj.plottingTools import plotTimeTrajectory
from zProj.lqrUtils import infiniteHorizonIntegralLqr


def controller(xDyn, xCtrl, xTrim, uTrim, Ci, Ki, Kp, r):
    x_fb = xDyn[:8]  # no position feedback
    dxCtrl = Ci @ (x_fb - xTrim) - r
    u = -Kp @ (x_fb - xTrim) - Ki @ xCtrl + uTrim
    return u, dxCtrl


def main():
    # User inputs
    uvwTrim = np.zeros(3)
    Q = np.eye(8)
    R = np.eye(4)
    Qi = np.eye(4)
    Ci = np.zeros((4, 8))
    Ci[:, [0, 1, 2, 5]] = np.eye(4)
    xDyn0 = np.zeros(12)
    xDyn0[0:3] = 0
    xCtrl0 = np.zeros(4)
    T = 30
    dt = 0.1
    r = np.array([1, 1, 1, 0.3])

    # Get linearized system
    ac = Quadcopter()
    xTrim, uTrim = ac.trim(uvwTrim)
    A, B = ac.linearize(xTrim, uTrim)

    # Design LQR controller
    Ki, Kp = infiniteHorizonIntegralLqr(A, B, Q, R, Qi, Ci)

    # Simple Simulation
    dyn_fun = lambda t, x, u: ac.inertialDynamics(x, u)
    control_fun = lambda t, xDyn, xCtrl: controller(xDyn, xCtrl, xTrim, uTrim, Ci, Ki, Kp, r)
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)
    sim = Simulator(dyn_fun, control_fun, t_span, xDyn0, xCtrl0, t_eval=t_eval)
    tArr, xDynArr, xCtrlArr, uArr = sim.simulate()

    # Plot Results
    plotTimeTrajectory(tArr, xDynArr[:, 0:3], names=['u', 'v', 'w'], title="Body Velocities")
    plotTimeTrajectory(tArr, xDynArr[:, 3:6], names=['p', 'q', 'r'], title="Body Rates")
    plotTimeTrajectory(tArr, xDynArr[:, 6:9], names=['phi', 'theta', 'psi'], title="Euler Angles")
    plotTimeTrajectory(tArr, xDynArr[:, 9:12], names=['x', 'y', 'z'], title="Positions")
    plotTimeTrajectory(tArr, xCtrlArr, names=["u", "v", "w", "r"], title="Controller Integral States")
    plotTimeTrajectory(tArr, uArr, names=["thrust", "pDot", "qDot", "rDot"], title="Pseudo Controls")
    plt.show()


if __name__ == "__main__":
    main()
