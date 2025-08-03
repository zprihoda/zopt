import numpy as np
import scipy.integrate as spi
from typing import Callable


class Simulator():

    def __init__(
        self,
        dyn_fun: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        control_fun: Callable[[float, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        t_span: tuple[float, float],
        xDyn0: np.ndarray,
        xCtrl0: np.ndarray,
        method: str = "RK45",
        t_eval: np.ndarray | None = None
    ):
        """
        Initialize a Simulator object

        Arguments
        ---------
            dyn_fun: Dynamics function of the form `dx = f(t,x,u)`
            control_fun: Control Function of the form `(u,dxCtrl) = f(t,xDyn,xCtrl)`
            t_span: Start and end time to simulate
            xDyn0: Initial dynamic state
            xCtrl0: Initial controller state
            method: Method of solve_ivp to use
            t_eval: Times at which to store the computed solution
        """
        self.dyn_fun = dyn_fun
        self.control_fun = control_fun
        self.t_span = t_span
        self.xDyn0 = xDyn0
        self.xCtrl0 = xCtrl0
        self.nxDyn = len(xDyn0)
        self.method = method
        self.t_eval = t_eval

    def _getStates(self, z):
        xDyn = z[:self.nxDyn]
        xCtrl = z[self.nxDyn:]
        return xDyn, xCtrl

    def _step_fun(self, t, z):
        xDyn, xCtrl = self._getStates(z)
        u, dxCtrl = self.control_fun(t, xDyn, xCtrl)
        dxDyn = self.dyn_fun(t, xDyn, u)
        dz = np.concatenate([dxDyn, dxCtrl])
        return dz

    def simulate(self):
        """
        Run the simulation

        Returns
        -------
            tArr: Array of times
            xDynArr: Array of dynamic states
            xCtrlArr: Array of controller states
            uArr: Array of controls
        """
        z0 = np.concatenate([self.xDyn0, self.xCtrl0])
        out = spi.solve_ivp(self._step_fun, self.t_span, z0, method=self.method, t_eval=self.t_eval)
        tArr = out.t
        zArr = out.y.T
        xDynArr = zArr[:, :self.nxDyn]
        xCtrlArr = zArr[:, self.nxDyn:]
        uArr = np.array([self.control_fun(t, xDyn, xCtrl)[0] for (t, xDyn, xCtrl) in zip(tArr, xDynArr, xCtrlArr)])
        return tArr, xDynArr, xCtrlArr, uArr
