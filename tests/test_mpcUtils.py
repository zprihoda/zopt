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
