import numpy as np
import scipy.integrate as spi
from typing import Callable


class Simulator():

    def __init__(
        self,
        dyn_fun: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        control_fun: Callable[[float, np.ndarray], np.ndarray],
        t_span: tuple[float, float],
        x0: np.ndarray,
        method: str = "RK45",
        t_eval: np.ndarray | None = None
    ):
        """
        Initialize a Simulator object

        Arguments
        ---------
            dyn_fun: Dynamics function of the form `dx = f(t,x,u)`
            control_fun: Control Function of the form `u = f(t,x)`
            t_span: Start and end time to simulate
            x0: Initial state
            method: Method of solve_ivp to use
            t_eval: Times at which to store the computed solution
        """
        self.dyn_fun = dyn_fun
        self.control_fun = control_fun
        self.t_span = t_span
        self.x0 = x0
        self.method = method
        self.t_eval = t_eval

    def _step_fun(self, t, x):
        u = self.control_fun(t, x)
        dx = self.dyn_fun(t, x, u)
        return dx

    def simulate(self):
        """
        Run the simulation

        Returns
        -------
            tArr: Array of times
            xArr: Array of states
            uArr: Array of controls
        """
        out = spi.solve_ivp(self._step_fun, self.t_span, self.x0, method=self.method, t_eval=self.t_eval)
        tArr = out.t
        xArr = out.y.T
        uArr = np.array([self.control_fun(t, x) for (t, x) in zip(tArr, xArr)])
        return tArr, xArr, uArr
