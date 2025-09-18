import jax
import matplotlib.pyplot as plt
import numpy as np

from zopt.mpcUtils import lqrMpc
from zopt.quadcopter import Quadcopter


def main():
    x0 = np.zeros(12)
    x0[9:12] = np.array([10, 10, 10])
    dt = 0.1
    N = 50  # MPC horizon
    Q = np.eye(12)
    R = np.eye(4)
    tf = 20
    x_ub = np.array([1, 1, 1, 0.3, 0.3, 0.1, 0.5, 0.5, np.inf, np.inf, np.inf, np.inf])
    x_lb = -x_ub
    u_ub = np.array([3, 3, 3, 3])
    u_lb = np.array([-3, -3, -3, -3])

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
    # tol = 1e-6
    for i in range(N_t):
        # x = np.clip(x, x_lb + tol, x_ub + tol)
        u, traj, status = mpcProb.solve(x)
        xMpc[i] = traj.xTraj
        uMpc[i] = traj.uTraj
        # x = x + dt*np.asarray(ac.inertialDynamics(x, u+uTrim))
        # x = A@x + B@u
        x = xMpc[i][1]

    # plot results
    fig, axs = plt.subplots(3, 1, sharex=True)
    t_arr = np.arange(N_t) * dt
    tMpc = np.arange(N_t + N + 1) * dt

    # Plot mpc trajectories
    for i in range(N_t):
        for j in range(3):
            axs[j].plot(tMpc[i:i + N + 1], xMpc[i, :, 9 + j], alpha=0.1, color="blue")

    # Plot full trajectories
    for j in range(3):
        axs[j].plot(t_arr, xMpc[:, 0, 9 + j], color="blue")
        axs[j].grid()
    axs[0].set_ylabel("x")
    axs[1].set_ylabel("y")
    axs[2].set_ylabel("z")
    axs[0].set_title("Inertial Positions")
    plt.show()


if __name__ == "__main__":
    main()
