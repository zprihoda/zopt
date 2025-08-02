import matplotlib.pyplot as plt
import numpy as np

from zProj.quadcopter import Quadcopter
from zProj.simulator import Simulator
from zProj.plottingTools import plotTimeTrajectories
from zProj.lqrUtils import infiniteHorizonLqr, proportionalFeedbackController


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
    K = infiniteHorizonLqr(A, B, Q, R)

    # Simple Simulation
    dyn_fun = lambda t, x, u: ac.inertialDynamics(x, u)
    control_fun = lambda t, x: proportionalFeedbackController(x[:8], xTrim, uTrim, K)
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)
    sim = Simulator(dyn_fun, control_fun, t_span, x0, t_eval=t_eval)
    tArr, xArr, uArr = sim.simulate()

    # Plot Results
    stateGroupNames = ["Body Velocities (m/s)", "Body Rates (rad/s)", "Euler Angles (rad)", "Positions (m)"]
    stateGroups = [['u', 'v', 'w'], ['p', 'q', 'r'], ['phi', 'theta', 'psi'], ['x', 'y', 'z']]
    controlGroupNames = ["Accel Commands (m/s^2, rad/s^2)"]
    controlGroups = [['thrust', 'pDot', 'qDot', 'rDot']]
    plotTimeTrajectories(tArr, xArr, uArr, stateGroupNames, stateGroups, controlGroupNames, controlGroups)
    plt.show()


if __name__ == "__main__":
    main()
