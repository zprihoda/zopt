import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple


class Trajectory(NamedTuple):
    """Trajectory tuple: (xTraj, uTraj)"""
    xTraj: jnp.ndarray
    uTraj: jnp.ndarray

    def __getitem__(self, k: int):
        return jax.tree.map(lambda x: x[k], self)


class CostFunction(NamedTuple):
    """
    Cost function tuple: (runningCost, terminalCost)
    ```
    J = terminalCost(x[N]) + sum(runningCost(x[i],u[i]))
    ```
    """
    runningCost: Callable[[jnp.ndarray, jnp.ndarray], float]
    terminalCost: Callable[[jnp.ndarray], float]

    @classmethod
    def runningOnly(cls, runningCost: Callable[[jnp.ndarray, jnp.ndarray], float], m: int = 1):
        """
        Construct a cost function with the same terminal and running cost functions
        `terminalCost(xf) = runningCost(xf,0)`

        Arguments
        ---------
            runningCost : Running cost function: `j = runningCost(x,u)`
            m : Dimension of u, used to size the 0 vector for the terminal cost.
                Defaults to 1, which may work with many cost functions via array broadcasting
        """
        terminalCost = lambda x: runningCost(x, jnp.zeros(m))
        return cls(runningCost, terminalCost)

    def __call__(self, traj: Trajectory, k=None):
        """
        Evaluate cost function for  a given trajectory

        Arguments
        ---------
            traj : Trajectory to evaluate. Dimensions must be: `xTraj = (N+1,n), uTraj = (N,m)`
            k : Optional index. If provided, only evaluate the running cost at index k of the trajectory
        """
        runningCost, terminalCost = self
        xTraj, uTraj = traj
        if k is None:
            J = jnp.sum(jax.vmap(runningCost)(xTraj[:-1], uTraj)) + terminalCost(xTraj[-1])
        else:
            J = runningCost(xTraj[k], uTraj[k])
        return J


class QuadraticValueFunction(NamedTuple):
    """
    Quadratic value function tuple: (v, v_x, v_xx)
    Representing a 2nd order Taylor series expansion: `V(x) = v + v_x.T @ x + 0.5 * (x.T @ v_xx @ x)`
    """
    v: jnp.ndarray
    v_x: jnp.ndarray
    v_xx: jnp.ndarray

    def __call__(self, x: jnp.ndarray):
        v, v_x, v_xx = self
        return v + v_x.T @ x + 0.5 * x.T @ v_xx @ x

    @classmethod
    def fromTerminalCostFunction(cls, costFun: CostFunction, xf: jnp.array):
        """
        Construct a quadratic value function from a terminal cost function and final state
        Can be used to initialize for Riccati recursion
        """
        cf = costFun.terminalCost
        v = cf(xf)
        v_x = jax.grad(cf)(xf)
        v_xx = jax.hessian(cf)(xf)
        return cls(v, v_x, v_xx)


class QuadraticCostFunction(NamedTuple):
    """
    Quadratic cost tuple: (c, c_x, c_u, c_xx, c_ux, c_uu)
    Representing a 2nd order Taylor series expansion:
    ```
    C(x,u) = c + c_x.T @ x + c_u.T @ u + 0.5 * (x.T @ c_xx @ x + 2*u.T @ c_ux @ x + u.T @ c_uu @ u)
    ```
    """
    c: jnp.ndarray
    c_x: jnp.ndarray
    c_u: jnp.ndarray
    c_xx: jnp.ndarray
    c_ux: jnp.ndarray
    c_uu: jnp.ndarray

    @classmethod
    def from_function(cls, costFun: CostFunction, x0: jnp.ndarray, u0: jnp.ndarray):
        """Initialize from a non-linear cost function `c(x,u)` about (x0,u0)"""
        runningCost = costFun.runningCost
        c = runningCost(x0, u0)
        c_x, c_u = jax.jacobian(runningCost, argnums=(0, 1))(x0, u0)
        ((c_xx, _), (c_ux, c_uu)) = jax.hessian(runningCost, (0, 1))(x0, u0)
        return cls(c, c_x, c_u, c_xx, c_ux, c_uu)

    @classmethod
    def from_trajectory(cls, costFun: CostFunction, traj: Trajectory):
        """
        Initialize from a non-linear cost function `c(x,u)` about trajectory `(xTraj,uTraj)`
        Trajectory shapes must be: `xTraj = (N+1,n), uTraj = (N,m)`
        """
        xTraj, uTraj = traj
        return jax.vmap(lambda x0, u0: cls.from_function(costFun, x0, u0))(xTraj[:-1], uTraj)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, k: int = None):
        """Evaluate cost function at (x,u)"""
        c, c_x, c_u, c_xx, c_ux, c_uu = self
        if k is None and c.ndim != 0:
            raise ValueError("Must specify index for multi-dimensional cost")
        return c + c_x @ x + c_u @ u + 0.5 * (x.T @ c_xx @ x + 2 * u.T @ c_ux @ x +
                                              u.T @ c_uu @ u) if k is None else self[k](x, u)

    def __getitem__(self, k: int):
        return jax.tree.map(lambda x: x[k], self)


class AffineDynamics(NamedTuple):
    """
    Discrete affine dynamics tuple: (f, f_x, f_u)
    1st order taylor series expansion: `xOut = f + f_x @ x + f_u @ u`
    """
    f: jnp.ndarray
    f_x: jnp.ndarray
    f_u: jnp.ndarray

    @classmethod
    def from_function(cls, dynFun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], x0: jnp.ndarray, u0: jnp.ndarray):
        """First order Taylor series expansion of the dynamics function `f(x,u)` about (x0,u0)"""
        f = dynFun(x0, u0)
        f_x = jax.jacobian(dynFun, 0)(x0, u0)
        f_u = jax.jacobian(dynFun, 1)(x0, u0)
        return cls(f, f_x, f_u)

    @classmethod
    def from_trajectory(cls, dynFun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], traj: Trajectory):
        """
        Second order Taylor series expansion of cost function `c(x,u)` about (xTraj,uTraj)
        Dimensions must be: `xTraj = (N+1,n), uTraj = (N,m)`
        """
        xTraj, uTraj = traj
        return jax.vmap(lambda x0, u0: cls.from_function(dynFun, x0, u0))(xTraj[:-1], uTraj)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, k: int = None):
        f, f_x, f_u = self
        if k is None and f.ndim != 1:
            raise ValueError("Must specify index for multi-dimensional dynamics")
        return f + f_x @ x + f_u @ u if k is None else self[k](x, u)

    def __getitem__(self, k: int):
        return jax.tree.map(lambda x: x[k], self)


class QuadraticDynamics(NamedTuple):
    """
    Quadratic dynamics tuple (f, f_x, f_u, f_xx, f_ux, f_uu)
    ```
    f(x,u) = f + f_x @ x + f_u @ u + 0.5 * (x.T @ f_xx @ x + 2*u.T @ f_ux @ x + u.T @ f_uu @ u)
    ```
    """
    f: jnp.ndarray
    f_x: jnp.ndarray
    f_u: jnp.ndarray
    f_xx: jnp.ndarray
    f_ux: jnp.ndarray
    f_uu: jnp.ndarray

    @classmethod
    def from_function(cls, dynFun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], x0: jnp.ndarray, u0: jnp.ndarray):
        """Second order Taylor series expansion of dynamics function `f(x,u)` about (x0,u0)"""
        f = dynFun(x0, u0)
        f_x, f_u = jax.jacobian(dynFun, argnums=(0, 1))(x0, u0)
        ((f_xx, _), (f_ux, f_uu)) = jax.hessian(dynFun, (0, 1))(x0, u0)
        return cls(f, f_x, f_u, f_xx, f_ux, f_uu)

    @classmethod
    def from_trajectory(cls, dynFun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], traj: Trajectory):
        """
        Second order Taylor series expansion of dynamics function `f(x,u)` about (xTraj,uTraj)
        Dimensions must be: `xTraj = (N+1,n), uTraj = (N,m)`
        """
        xTraj, uTraj = traj
        return jax.vmap(lambda x0, u0: cls.from_function(dynFun, x0, u0))(xTraj[:-1], uTraj)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, k: int = None):
        f, f_x, f_u, f_xx, f_ux, f_uu = self
        if k is None and f.ndim != 1:
            raise ValueError("Must specify index for trajectories")
        return f + f_x @ x + f_u @ u + 0.5 * (x.T @ f_xx @ x + 2 * u.T @ f_ux @ x +
                                              u.T @ f_uu @ u) if k is None else self[k](x, u)

    def __getitem__(self, k: int):
        return jax.tree.map(lambda x: x[k], self)


class AffinePolicy(NamedTuple):
    """
    Affine control policy tuple: (l, L):
    `u = l + L @ x`
    """
    l: jnp.ndarray
    L: jnp.ndarray

    def __call__(self, x: jnp.ndarray, k: int = None, alpha: float = 1):
        """Evaluate control policy for given x at time step k with optional affine scaling factor alpha"""
        l, L = self
        if k is None and l.ndim != 1:
            raise ValueError("Must specify index for multi-dimensional policy")
        return alpha * l + L @ x if k is None else self[k](x, alpha=alpha)

    def __getitem__(self, k: int):
        return jax.tree.map(lambda x: x[k], self)


class QuadraticDeltaCost(NamedTuple):
    """
    Quadratic delta cost function for line search approximations
    dJ_exp = alpha * dJ_lin + alpha**2 * dJ_quad
    """
    dJ_lin: float
    dJ_quad: float

    def __call__(self, alpha):
        dJ_lin, dJ_quad = self
        return alpha * (dJ_lin + alpha * dJ_quad)
