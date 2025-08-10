import jax
import numpy as np
import warnings

from typing import Callable


## iLQR and DDP
class iLQR():
    """Iterative LQR Solver"""

    def __init__(
        self,
        dynFun: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
        costFun: Callable[[int, np.ndarray, np.ndarray], float],
        x0: np.ndarray,
        u: np.ndarray,
        terminalCostFun: Callable[[np.ndarray], float] = None,
        maxIter: int = 10,
        tol: float = 1e-3,
        jittable: bool = True
    ):
        """
        Iterative LQR constructor

        Arguments
        ---------
            dynFun : Discrete dynamics function of the form `xOut = f(k,x,u)`
            costFun : Discrete cost function of the form `j = c(k,x,u)`
            x0 : Initial state, array of shape (n,)
            u : Initial guess for control trajectory, array of shape (N,m)
            terminalCostFun : Terminal cost function of the form: `j = cf(x)`
                Default: cf(x) = c(N,x,0)
            maxIter : Maximum number of optimization iterations
                Default: 100
            tol : Convergence tolerance, will exit if norm(xPrev-x) <= tol
                Default 1e-3
            jittable : Specifies whether the dynamics and cost functions are compatible with jax.jit
        """
        self.x0 = x0
        self.u = u
        self.maxIter = maxIter
        self.tol = tol
        self._computeDerivatives(
            dynFun, costFun, terminalCostFun, jittable
        )  # Also stores dynamics and cost functions as instance parametersd

    def _computeDerivatives(self, f, c, cf, jittable):
        fx = jax.jacrev(f, 1)
        fu = jax.jacrev(f, 2)
        cx = jax.jacrev(c, 1)
        cu = jax.jacrev(c, 2)
        cxx = jax.jacfwd(cx, 1)
        cux = jax.jacfwd(cu, 1)
        cuu = jax.jacfwd(cu, 2)

        if cf is None:
            N, m = self.u.shape
            cf = lambda x: c(N, x, np.zeros(m))
            cfx = lambda x: cx(N, x, np.zeros(m))
            cfxx = lambda x: cxx(N, x, np.zeros(m))
        else:
            cfx = jax.jacrev(cf, 0)
            cfxx = jax.jacfwd(cfx, 0)

        if jittable:
            f = jax.jit(f)
            c = jax.jit(c)
            fx = jax.jit(fx)
            fu = jax.jit(fu)
            cx = jax.jit(cx)
            cu = jax.jit(cu)
            cxx = jax.jit(cxx)
            cux = jax.jit(cux)
            cuu = jax.jit(cuu)
            cf = jax.jit(cf)
            cfx = jax.jit(cfx)
            cfxx = jax.jit(cfxx)

        self.f = f
        self.c = c
        self.fx = fx
        self.fu = fu
        self.cx = cx
        self.cu = cu
        self.cxx = cxx
        self.cux = cux
        self.cuu = cuu
        self.cf = cf
        self.cfx = cfx
        self.cfxx = cfxx

    def solve(self) -> tuple[np.ndarray, np.ndarray, Callable[[int, np.ndarray], np.ndarray]]:
        """
        Solve the iLQR problem

        Returns
        -------
            xTraj : Optimal state trajectory of shape (N+1,n)
            uTraj : Optimal control trajectory of shape (N,m)
            Pi : Optimal control policy of the form u = Pi(k,dx)
                where `dx = x - xTraj[k]`
        """

        # Get various dimensions
        n = len(self.x0)
        N, m = self.u.shape

        # Initialize stuff
        xPrev = np.inf * np.ones((N + 1, n))
        u = self.u
        Pi = lambda k, dx: 0  # du = Pi(k,dx)
        converged = False

        # Main iLQR loop
        for iter in range(self.maxIter):
            x, u = self._forwardPass(u, xPrev, Pi)

            # Check convergence Criteria
            delta = np.linalg.norm(x - xPrev)
            if delta <= self.tol:
                converged = True
                break

            Pi = self._backwardPass(x, u)

            xPrev = x.copy()

        if not converged:
            warnings.warn("ILQR reached max iterations and did not converge. Most recent delta = {:.3g}".format(delta))

        return x, u, Pi

    def _forwardPass(self, u, xPrev, Pi):
        N, m = u.shape
        n = len(self.x0)
        x = np.zeros((N + 1, n))
        x[0] = self.x0

        for k in range(N):
            dx = x[k] - xPrev[k]
            du = Pi(k, dx)
            u[k] = u[k] + du
            x[k + 1] = self.f(k, x[k], u[k])

        return x, u

    def _backwardPass(self, x, u):
        N, m = u.shape
        n = x.shape[1]

        v = self.cf(x[N])
        vVec = self.cfx(x[N])
        V = self.cfxx(x[N])

        lArr = np.zeros((N, m))
        LArr = np.zeros((N, m, n))

        for k in range(N - 1, -1, -1):
            fxk = self.fx(x[k], u[k])
            fuk = self.fu(x[k], u[k])

            Q = self.c(x[k], u[k]) + v
            Qx = self.cx(x[k], u[k]) + fxk.T @ vVec
            Qu = self.cu(x[k], u[k]) + fuk.T @ vVec
            Qxx = self.cxx(x[k], u[k]) + fxk.T @ V @ fxk
            Quu = self.cuu(x[k], u[k]) + fuk.T @ V @ fuk
            Qux = self.cux(x[k], u[k]) + fuk.T @ V @ fxk

            l = -np.linalg.solve(Quu, Qu)
            L = -np.linalg.solve(Quu, Qux)

            v = Q - 0.5 * l.T @ Quu @ l
            vVec = Qx - L.T @ Quu @ l
            V = Qxx - L.T @ Quu @ L

            lArr[k] = l
            LArr[k] = L

        # Update control policy
        Pi = lambda dx, k: lArr[k] + LArr[k] @ dx

        return Pi
