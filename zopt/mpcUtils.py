import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

from zopt.pytrees import Trajectory


class lqrMpc():

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        N: int,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
        u_lb: np.ndarray,
        u_ub: np.ndarray,
        Qf: np.ndarray = None
    ):
        """
        Setup an LQR MPC problem

        Arguments
        ---------
            A : Dynamics matrix; shape = (n,n)
            B : Input matrix; shape = (n,m)
            Q : State cost matrix; shape = (n,n)
            R : Control cost matrix; shape = (m,m)
            N : MPC horizon
            x_lb : State lower bound
            x_ub : State upper bound
            u_lb : Control lower bound
            u_ub : Control upper bound
            Qf : Terminal cost matrix, optional; shape = (n,n)
                Defaults to Q
        """
        if Qf is None:
            Qf = Q

        # Setup cvx problem
        n, m = B.shape
        x = cvx.Variable((N + 1, n), name="x")
        u = cvx.Variable((N, m), name="u")
        x0 = cvx.Parameter(n, name="x0")
        runningCost = cvx.sum([cvx.quad_form(x[k], Q) + cvx.quad_form(u[k], R) for k in range(N)])
        terminalCost = cvx.quad_form(x[-1], Qf)
        cost = terminalCost + runningCost
        constr = [x[k + 1] == A @ x[k] + B @ u[k] for k in range(N)]
        constr += [x >= x_lb[None, :], x <= x_ub[None, :]]
        constr += [u >= u_lb[None, :], u <= u_ub[None, :]]
        constr += [x[0] == x0]
        self.prob = cvx.Problem(cvx.Minimize(cost), constr)

    def solve(self, x0: np.ndarray) -> tuple[np.ndarray, Trajectory, str]:
        """
        Solve the MPC step at state x0

        Arguments
        ---------
            x0 : Initial state

        Returns
        -------
            u : Optimal control at current time step
            traj : Trajectory tuple (xTraj, uTraj) containing full state and control trajectories
            status : cvx problem status; one of [optimal, infeasible, unbounded]
        """
        self.prob.param_dict['x0'].value = x0
        self.prob.solve(solver="OSQP")
        status = self.prob.status
        xTraj = self.prob.var_dict['x'].value
        uTraj = self.prob.var_dict['u'].value
        return uTraj[0], Trajectory(xTraj, uTraj), status


def plotMpcTrajectory(traj: np.ndarray,
                      dt: float,
                      names: list[str] = None,
                      title: str = None) -> tuple[plt.figure, plt.axes]:
    """
    Plot an mpc trajectory array

    Arguments
    ---------
        traj : Array of shape (N_t, N_mpc, n) such that:
            traj[i] = MPC trajectory at time step i of shape (N_mpc, n)
        dt : Time step
        names : List of n signal names
    """
    (N_t, N_mpc, n) = traj.shape

    if names is None:
        names = [f'x{i}' for i in range(n)]

    tNom = np.arange(N_t) * dt
    tMpc = np.arange(N_t + N_mpc) * dt

    # Plot each MPC trajectory
    fig, axs = plt.subplots(n, 1, sharex=True)
    for i in range(N_t):
        for j in range(n):
            axs[j].plot(tMpc[i:i + N_mpc], traj[i, :, j], alpha=0.1, color="blue")

    # Plot full trajectories
    for j in range(3):
        axs[j].plot(tNom, traj[:, 0, j], color="blue")
        axs[j].set_ylabel(names[j])
        axs[j].grid()
    axs[0].set_xlim([0, tNom[-1]])
    if title is not None:
        axs[0].set_title(title)
    return fig, axs
