import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from zProj.quadcopter import Quadcopter
from zProj.simulator import Simulator, SimBlock
from zProj.plottingTools import plotTimeTrajectory
from zProj.quadcopterAnimation import QuadcopterAnimation
from zProj.ilqrUtils import DDP


def cost(k, x, u, Q, R):
    return x.T @ Q @ x + u.T @ R @ u


def controller(k, x, xTraj, uTraj, LArr):
    return LArr[k] @ (x - xTraj[k]) + uTraj[k]


def main():
    # User inputs
    x0 = np.zeros(12)
    x0[9:12] = np.array([0, 5, 0])
    dt = 0.1
    N = 100
    Q = np.eye(12)
    R = 0.2*np.eye(4)
    tArr = np.arange(N + 1) * dt

    # Get quadcopter dynamics (and trim)
    ac = Quadcopter()
    _, uTrim = ac.trim(np.zeros(3))

    # Setup and solve DDP problem
    dynFun = lambda k, x, u: x + dt * ac.inertialDynamics(x, u)
    costFun = lambda k, x, u: cost(k, x, u, Q, R)
    terminalCostFun = lambda x: 10 * x.T @ np.eye(12) @ x
    uGuess = np.repeat(uTrim[None, :], N, axis=0)
    prob = DDP(dynFun, costFun, x0, uGuess, terminalCostFun=terminalCostFun)
    xTraj, uTraj, LArr = prob.solve()
    xTraj = jnp.asarray(xTraj)
    uTraj = jnp.asarray(uTraj)
    LArr = jnp.asarray(LArr)

    # Simple Simulation
    xCtrl0 = np.array([])
    noisyDynFun = lambda k, x, u: (None, x + dt * ac.inertialDynamics(x, u, wind_ned=np.array([3, 1, 0])))
    dynamicsBlock = SimBlock(noisyDynFun, x0, dt=dt, name="Dynamics")
    controllerBlock = SimBlock(
        lambda k, xCtrl, x: (controller(k, x, xTraj, uTraj, LArr), np.array([])),
        xCtrl0,
        dt=dt,
        name="Controller",
    )
    t_span = (0, tArr[-1])
    sim = Simulator([controllerBlock, dynamicsBlock], t_span)
    tSim, _, xSim, uSim, _ = sim.simulate()

    # Plot Results
    fig = plotTimeTrajectory(tArr, xTraj[:, 0:3], names=['u', 'v', 'w'], title="Body Velocities")
    plotTimeTrajectory(tSim, xSim[:, 0:3], fig=fig)
    plt.legend(["DDP Trajectory", "Sim"])
    fig = plotTimeTrajectory(tArr, xTraj[:, 3:6], names=['p', 'q', 'r'], title="Body Rates")
    plotTimeTrajectory(tSim, xSim[:, 3:6], fig=fig)
    plt.legend(["DDP Trajectory", "Sim"])
    fig = plotTimeTrajectory(tArr, xTraj[:, 6:9], names=['phi', 'theta', 'psi'], title="Euler Angles")
    plotTimeTrajectory(tSim, xSim[:, 6:9], fig=fig)
    plt.legend(["DDP Trajectory", "Sim"])
    fig = plotTimeTrajectory(tArr, xTraj[:, 9:12], names=['x', 'y', 'z'], title="Positions")
    plotTimeTrajectory(tSim, xSim[:, 9:12], fig=fig)
    plt.legend(["DDP Trajectory", "Sim"])
    fig = plotTimeTrajectory(tArr[:-1], uTraj, names=["thrust", "pDot", "qDot", "rDot"], title="Pseudo Controls")
    plotTimeTrajectory(tSim[:-1], uSim, fig=fig)
    plt.legend(["DDP Trajectory", "Sim"])
    plt.show()

    # Simulate Results
    animObj = QuadcopterAnimation(tSim, xSim)
    ani = animObj.animate()
    plt.show()


if __name__ == "__main__":
    main()
