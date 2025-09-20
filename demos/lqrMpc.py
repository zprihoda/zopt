import jax
import matplotlib.pyplot as plt
import numpy as np

from zopt.mpcUtils import lqrMpc, plotMpcTrajectory, animateMpcTrajectory
from zopt.quadcopter import Quadcopter


def main():
    x0 = np.zeros(12)
    x0[9:12] = np.array([10, 10, 10])
    dt = 0.1
    N = 25  # MPC horizon
    Q = np.eye(12)
    R = np.eye(4)
    tf = 20
    x_ub = np.array([1, 1, 1, 0.3, 0.3, 0.1, 0.5, 0.5, np.inf, np.inf, np.inf, np.inf])
    x_lb = -x_ub
    u_ub = np.array([3, 3, 3, 3])
    u_lb = -u_ub

    # Get quadcopter linear dynamics
    ac = Quadcopter()
    _, uTrim = ac.trim(np.zeros(3))
    Aw, Bw = jax.jacobian(ac.inertialDynamics, argnums=(0, 1))(np.zeros(12), uTrim)
    A = np.asarray(np.eye(12) + dt * Aw)
    B = np.asarray(dt * Bw)

    # Run mpc iterations
    N_t = int(tf / dt + 1)
    xMpc = np.zeros((N_t, N + 1, 12))
    uMpc = np.zeros((N_t, N, 4))
    x = x0

    mpcProb = lqrMpc(A, B, Q, R, N, x_lb, x_ub, u_lb, u_ub)
    tol = 1e-6
    for i in range(N_t):
        x = np.clip(x, x_lb + tol, x_ub - tol)
        u, traj, status = mpcProb.solve(x)
        xMpc[i] = traj.xTraj
        uMpc[i] = traj.uTraj
        x = xMpc[i][1]  # Assume perfect tracking

    # plot results
    plotMpcTrajectory(xMpc[:, :, 0:3], dt, names=['u', 'v', 'w'], title='Body Velocities')
    plotMpcTrajectory(xMpc[:, :, 3:6], dt, names=['p', 'q', 'r'], title='Body Rates')
    plotMpcTrajectory(xMpc[:, :, 6:9], dt, names=['phi', 'theta', 'psi'], title='Euler Angles')
    plotMpcTrajectory(xMpc[:, :, 9:12], dt, names=['x', 'y', 'z'], title='Positions')
    plotMpcTrajectory(uMpc, dt, names=['thrust', 'Mx', 'My', 'Mz'], title='Controls')
    plt.show()

    # Animate reuslts
    _ = animateMpcTrajectory(xMpc[:, :, 6:9], dt, names=['phi', 'theta', 'psi'], title='Euler Angles', speed=2)
    plt.show()


if __name__ == "__main__":
    main()
