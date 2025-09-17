import cvxpy as cvx
import numpy as np

from pytrees import Trajectory

class lqrMpc():
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, x_lb: np.ndarray, x_ub: np.ndarray, u_lb: np.ndaray, u_ub: np.ndarray, Qf: np.ndarray = None):
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
        n,m = B.shape()
        x = cvx.Variable((N+1, n))
        u = cvx.Variable((N, m))
        x0 = cvx.Parameter(n)
        runningCost = cvx.sum([cvx.quad_form(x[k], Q) + cvx.quad_form(u[k], R) for k in range(N)])
        terminalCost = cvx.quad_form(x[-1], Qf)
        objective = terminalCost + runningCost
        constr = [x[k+1] == A@x[k] + B@u[k] for k in range(N)]
        constr += [x >= x_lb, x <= x_ub]
        constr += [u >= u_lb, u <= u_ub]
        constr += [x[0] == x0]
        self.prob = cvx.Problem(objective, constr)

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
        self.prob.solve()
        status = self.prob.status
        xTraj = self.prob.var_dict['x'].value
        uTraj = self.prob.var_dict['u'].value
        return uTraj[0], Trajectory(xTraj, uTraj), status
