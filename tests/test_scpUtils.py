import numpy as np
import pytest

from zopt.pytrees import Trajectory
from zopt.scpUtils import sequentialConvexProgramming

@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_sequentialConvexProgramming():
    A = np.eye(2)
    B = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    N = 3
    dt = 0.1
    x0 = np.ones(2)
    xTraj0 = np.ones((N + 1, 2))
    uTraj0 = np.zeros((N, 2))
    traj0 = Trajectory(xTraj0, uTraj0)

    f = lambda x, u: A @ x + B @ u
    runningCost = lambda x, u: x @ Q @ x + u @ R @ u
    terminalCost = lambda x: x @ Q @ x

    traj, converged = sequentialConvexProgramming(traj0, x0, f, runningCost, terminalCost, dt)
    assert converged
