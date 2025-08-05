import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate as spi

from typing import Callable


class SimBlock():

    def __init__(
        self,
        fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x0: np.ndarray,
        dt: float = 0,
        jittable: bool = True,
        name: str = None
    ):
        """
        Create a simulation block

        Arguments
        ---------
        fun : Update function of the form:
            - Continuous: `y,xDot = fun(x,u)`
            - Discrete: `y,x[k+1] = fun(x[k],u)`
        x0 : Initial block state
        dt : Sample time of block, 0=continuous
        jittable : Flag specifying whether fun is compatible with jax jit compilation
        """
        if jittable:
            fun = jax.jit(fun)

        self.update = fun
        self.dt = dt
        self.jittable = jittable
        self.x0 = x0
        self.nx = len(x0)
        self.name = name


# TODO: Add signal map to customize input/output mapping between blocks
#   Currently assume all outputs of block 0 are inputs to block 1 and vice-versa
class SignalMap:
    pass


class Simulator():

    def __init__(
        self,
        blocks: list[SimBlock],
        t_span: tuple[float, float],
        method: str = "RK45",
        t_eval: np.ndarray | None = None
    ):
        """
        Initialize a Simulator object

        Arguments
        ---------
            blocks: List of simBlocks to use. Currently only supports 2 blocks
            t_span: Start and end time to simulate
            method: Method of solve_ivp to use
            t_eval: Times at which to store the computed solution
        """
        self.blocks = blocks
        self.t_span = t_span
        self.method = method
        self.t_eval = t_eval
        self.dt = self._getSampleTime()
        self._step_fun = self._buildStepFunction()

    def _getSampleTime(self):
        assert len(set([block.dt for block in self.blocks])) == 1, "Multi-sample time not implemented yet."
        dt = self.blocks[0].dt
        return dt

    def _buildStepFunction(self):
        assert len(self.blocks) == 2, "Currently only supports 2 simBlocks."

        jittable = all([block.jittable for block in self.blocks])
        if self.dt == 0:
            stepFun = lambda t, x: self._internalStepFunContinuous(t, x)
        else:
            stepFun = lambda k, x: self._internalStepFunDiscrete(k, x)

        if jittable:
            stepFun = jax.jit(stepFun)

        return stepFun

    def _getStates(self, x):
        x0 = x[:self.blocks[0].nx]
        x1 = x[self.blocks[0].nx:]
        return x0, x1

    def _internalStepFunContinuous(self, t, x):
        """internal step function for continuous time sims
        TODO:
        1. Once signal map implemented, replace x1 input to block[0] with y1
        2. Figure out how to keep y1 in memory and still work with jax.jit?
            - may have to write a custom ivp solver (or checkout open source alternatives)
        """
        x0, x1 = self._getStates(x)
        y0, dx0 = self.blocks[0].update(t, x0, x1)  # TODO: see above notes 1,2
        y1, dx1 = self.blocks[1].update(t, x1, y0)
        dx = jnp.concatenate([dx0, dx1])
        return dx

    def _internalStepFunDiscrete(self, k, x):
        raise NotImplementedError("Discrete time step function not implemented")

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
        x0 = np.concatenate([block.x0 for block in self.blocks])

        if self.dt == 0:
            out = spi.solve_ivp(self._step_fun, self.t_span, x0, method=self.method, t_eval=self.t_eval)
        else:
            raise NotImplementedError("Discrete time sim not implemented")

        tArr = out.t
        xArr = out.y.T
        x0Arr = xArr[:, :self.blocks[0].nx]
        x1Arr = xArr[:, self.blocks[0].nx:]
        y0Arr = np.array([self.blocks[0].update(t, x0, x1)[0] for (t, x0, x1) in zip(tArr, x0Arr, x1Arr)])
        y1Arr = np.array([self.blocks[1].update(t, x1, y0)[0] for (t, x1, y0) in zip(tArr, x1Arr, y0Arr)])
        return tArr, x0Arr, x1Arr, y0Arr, y1Arr
