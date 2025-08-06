import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi

from zProj.quadcopter import Quadcopter
from zProj.simulator import Simulator, SimBlock
from zProj.plottingTools import plotTimeTrajectory
from zProj.lqrUtils import infiniteHorizonIntegralLqr


def getOpenLoopTrajectory(
    A: np.ndarray,
    B: np.ndarray,
    xTrim: np.ndarray,
    uTrim: np.ndarray,
    T: float,
    dt: float,
    x0: np.ndarray,
    xf: np.ndarray
):
    """Design a trajectory using cvxpy and linearized inertial dynamics about hover"""
    nx, nu = B.shape
    tTraj = np.arange(0, T, step=dt)
    nt = len(tTraj)

    # Setup and solve CVX problem
    xTraj = cvx.Variable((nt, nx))
    du = cvx.Variable((nt, nu))
    du_dt_max = dt * np.array([1, 1, 1, 1])  # Limit rate of change of control variables
    objective = cvx.Minimize(cvx.sum(cvx.norm(du, axis=1)))
    constraints = [xTraj[0] == x0, xTraj[-1] == xf]
    constraints += [xTraj[i + 1] == xTraj[i] + dt * (A @ xTraj[i] + B @ du[i]) for i in range(nt - 1)]
    constraints += [du[0] == 0]
    constraints += [cvx.abs(du[i + 1] - du[i]) <= du_dt_max for i in range(nt - 1)]
    prob = cvx.Problem(objective, constraints)
    prob.solve()

    if prob.status != "optimal":
        raise RuntimeError("CVX failed to converge!")

    # Use interp to convert to continuous functions
    x = xTraj.value.T
    u = (du.value + uTrim).T
    t = jnp.concatenate([jnp.array([-1]), tTraj, jnp.array([np.inf])])
    x = jnp.concatenate([x[:,[0]], x, x[:,[-1]]], axis=1)
    u = jnp.concatenate([u[:,[0]], u, u[:,[-1]]], axis=1)
    xFun = lambda tq: jaxInterp(t, x, tq)
    uFun = lambda tq: jaxInterp(t, u, tq)

    return xFun, uFun

def jaxInterp(x, y, xq):
    """Jax compliant linear vector interpolation"""
    idx = jnp.searchsorted(x, xq) - 1
    frac = (xq - x[idx]) / (x[idx + 1] - x[idx])
    yq = (1 - frac) * y[:, idx] + frac * y[:, idx + 1]
    return yq


def controller(t, xDyn, xCtrl, xTraj, uTraj, Ci, Ki, Kp):
    dxCtrl = Ci @ (xDyn - xTraj(t))
    u = -Kp @ (xDyn - xTraj(t)) - Ki @ xCtrl + uTraj(t)
    return u, dxCtrl


def main():
    # User inputs
    T = 30
    dt = 0.1
    Q = np.eye(12)
    R = np.eye(4)
    Qi = np.eye(3)
    Ci = np.zeros((3, 12))
    Ci[:, 9:12] = np.eye(3)
    xDyn0 = np.zeros(12)
    xCtrl0 = np.zeros(3)
    xf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 5, 5])
    T = 10
    dt = 0.1

    # Trim and linearize inertial dynamics
    ac = Quadcopter()
    xTrim, uTrim = ac.trim(np.zeros(3))
    xTrim = np.concatenate([xTrim, np.zeros(4)])  # Append inertial states
    A, B = jax.jacobian(ac.inertialDynamics, argnums=(0, 1))(xTrim, uTrim)

    # Get open loop trajectory
    xTraj, uTraj = getOpenLoopTrajectory(A, B, xTrim, uTrim, T, dt, xDyn0, xf)

    # Design LQR controller
    # NOTE: Could linearize and solve LQR about the open loop trajectory instead.
    #   We'll cover this in the iLQR demo later
    Ki, Kp = infiniteHorizonIntegralLqr(A, B, Q, R, Qi, Ci)

    # Run Simulation
    dynamics = SimBlock(lambda t, x, u: (None, ac.inertialDynamics(x, u)), xDyn0, name="Dynamics")
    controlBlock = SimBlock(
        lambda t, xCtrl, xDyn: controller(t, xDyn, xCtrl, xTraj, uTraj, Ci, Ki, Kp),
        xCtrl0,
        name="Controller"
    )
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)
    sim = Simulator([controlBlock, dynamics], t_span, t_eval=t_eval)
    tArr, xCtrlArr, xDynArr, uArr, _ = sim.simulate()

    # Plot Results
    xTrajArr = xTraj(tArr).T
    uTrajArr = uTraj(tArr).T
    fig = plotTimeTrajectory(tArr, xDynArr[:, 0:3], names=['u', 'v', 'w'], title="Body Velocities")
    plotTimeTrajectory(tArr, xTrajArr[:, 0:3], fig=fig)
    plt.legend(["Simulated", "Design Trajectory"])
    fig = plotTimeTrajectory(tArr, xDynArr[:, 3:6], names=['p', 'q', 'r'], title="Body Rates")
    plotTimeTrajectory(tArr, xTrajArr[:, 3:6], fig=fig)
    plt.legend(["Simulated", "Design Trajectory"])
    fig = plotTimeTrajectory(tArr, xDynArr[:, 6:9], names=['phi', 'theta', 'psi'], title="Euler Angles")
    plotTimeTrajectory(tArr, xTrajArr[:, 6:9], fig=fig)
    plt.legend(["Simulated", "Design Trajectory"])
    fig = plotTimeTrajectory(tArr, xDynArr[:, 9:12], names=['x', 'y', 'z'], title="Positions")
    plotTimeTrajectory(tArr, xTrajArr[:, 9:12], fig=fig)
    plt.legend(["Simulated", "Design Trajectory"])
    fig = plotTimeTrajectory(tArr, uArr, names=["thrust", "pDot", "qDot", "rDot"], title="Pseudo Controls")
    plotTimeTrajectory(tArr, uTrajArr, fig=fig)
    plt.legend(["Simulated", "Design Trajectory"])
    plotTimeTrajectory(tArr, xCtrlArr, names=["x", "y", "z"], title="Controller Integral State")

    plt.show()


if __name__ == "__main__":
    main()
