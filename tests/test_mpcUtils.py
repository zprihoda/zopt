import matplotlib.pyplot as plt
import numpy as np
import pytest

import zopt.mpcUtils as mpc


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_lqrMpc():
    A = np.eye(2)
    B = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    x_ub = np.ones(2)
    x_lb = -np.ones(2)
    u_ub = np.ones(2)
    u_lb = -np.ones(2)
    N = 2
    x0 = np.ones(2)

    prob = mpc.lqrMpc(A, B, Q, R, N, x_lb, x_ub, u_lb, u_ub)
    u, traj, status = prob.solve(x0)
    assert status == "optimal"


def test_plotMpcTrajectory():
    N_t = 2
    N_mpc = 3
    n = 4
    dt = 0.1
    traj = np.zeros((N_t, N_mpc, n))

    fig, ax = mpc.plotMpcTrajectory(traj, dt)
    plt.close(fig)


def test_plotMpcTrajectory2():
    N_t = 2
    N_mpc = 3
    n = 4
    dt = 0.1
    traj = np.zeros((N_t, N_mpc, n))

    fig, ax = mpc.plotMpcTrajectory(traj, dt, names=['a', 'b', 'c', 'd'], title="foobar")
    plt.close(fig)
