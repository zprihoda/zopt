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
        muMin: float = 1e-6,
        delta0: float = 2,
        maxIter: int = 100,
        tol: float = 1e-6,
        maxLineSearchIter: int = 16,
        betaLineSearch: float = 0.5,
        cLineSearch: float = 0.8,
        jittable: bool = True
    ):
        """
        Iterative LQR constructor

        Arguments
        ---------
        dynFun : Callable[[int, np.ndarray, np.ndarray], np.ndarray]
            Discrete dynamics function of the form `xOut = f(k,x,u)`
        costFun : Callable[[int, np.ndarray, np.ndarray], float]
            Discrete cost function of the form `j = c(k,x,u)`
        x0 : np.ndarray
            Initial state, array of shape (n,)
        u : np.ndarray,
            Initial guess for control trajectory, array of shape (N,m)
        terminalCostFun : Callable[[np.ndarray], float], optional
            Terminal cost function of the form: `j = cf(x)`
            Default: cf(x) = c(N,x,0)
        muMin : float
            Minimum cost regularization term, larger mu biases each step towards the current trajectory
            Default: 1e-6
        delta0 : float
            Minimal modification factor
            Default: 2
        maxIter : int, optional
            Maximum number of optimization iterations
            Default: 100
        tol : float, optional
            Convergence tolerance, will exit if norm(xPrev-x) <= tol
            Default: 1e-3
        maxLineSearchIter : int, optional
            Maximum number of line search iterations in forward pass
            Default: 16
        betaLineSearch : float, optional
            Scale factor per iteration of line search.
            Must be between 0 and 1
            Default: 0.5
        cLineSearch : float, optional
            Line search cost degredation threshold
            Must be between 0 and 1
            Default: 0.8
        jittable : bool, optional
            Specifies whether the dynamics and cost functions are compatible with jax.jit
            Default: True

        Notes
        -----
        Iterative lqr is very dependent on the initial guess.
        If the algorithm is failing to converge, consider providing a different starting point.
        """
        self.x0 = x0
        self.u = u
        self.tol = tol
        self.muMin = muMin
        self.delta0 = delta0
        self.maxIter = maxIter
        self.maxLineSearchIter = maxLineSearchIter
        self.betaLineSearch = betaLineSearch
        self.cLineSearch = cLineSearch
        self.jittable = jittable
        self._computeDerivatives(
            dynFun, costFun, terminalCostFun
        )  # Also stores dynamics and cost functions as instance parameters

    def _computeDerivatives(self, f, c, cf):
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

        if self.jittable:
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

    def _computeQ(self, k, x, u, V, vVec, v, mu):
        n = len(x)
        fxk = self.fx(k, x, u)
        fuk = self.fu(k, x, u)
        Vw = V + mu * np.eye(n)

        Q = self.c(k, x, u) + v
        Qx = self.cx(k, x, u) + fxk.T @ vVec
        Qu = self.cu(k, x, u) + fuk.T @ vVec
        Qxx = self.cxx(k, x, u) + fxk.T @ V @ fxk
        Quu = self.cuu(k, x, u) + fuk.T @ Vw @ fuk
        Qux = self.cux(k, x, u) + fuk.T @ Vw @ fxk
        return Q, Qx, Qu, Qxx, Quu, Qux

    def solve(self) -> tuple[np.ndarray, np.ndarray, Callable[[int, np.ndarray], np.ndarray]]:
        """
        Solve the iLQR problem

        Returns
        -------
        xTraj : np.ndarray
            Optimal state trajectory of shape (N+1,n)
        uTraj : np.ndarray
            Optimal control trajectory of shape (N,m)
        LArr : np.ndarray
            Optimal control gains of the form u = LArr[k] @ (x - xTraj[k])
        """

        # Get various dimensions
        n = len(self.x0)
        N, m = self.u.shape

        # Initialize stuff
        xPrev = np.inf * np.ones((N + 1, n))
        u = self.u
        LArr = np.zeros((N, m, n))
        lArr = np.zeros((N, m))
        converged = False
        mu = self.muMin
        delta = self.delta0

        J = np.inf
        dJ_lin = -1
        dJ_quad = 0

        # Main iLQR loop
        for iter in range(self.maxIter):
            init = (iter == 0)
            x, u, J = self._forwardPass(u, xPrev, LArr, lArr, init, J, dJ_lin, dJ_quad)

            # Check convergence Criteria
            delta_x = np.linalg.norm(x - xPrev)
            if delta_x <= self.tol:
                converged = True
                break

            LArr, lArr, mu, delta, dJ_lin, dJ_quad = self._backwardPass(x, u, mu, delta)
            xPrev = x.copy()

        if not converged:
            warnings.warn(
                "ILQR reached max iterations and did not converge. Most recent delta = {:.3g}".format(delta_x)
            )

        return x, u, LArr

    def _forwardPass(self, uPrev, xPrev, LArr, lArr, init, J_prev, dJ_lin, dJ_quad):
        N, m = uPrev.shape
        n = len(self.x0)
        x = np.zeros((N + 1, n))
        x[0] = self.x0

        alpha = 1
        converged = False
        for lineSearchIter in range(self.maxLineSearchIter):
            u = uPrev.copy()
            dJ_exp = alpha * dJ_lin + alpha**2 * dJ_quad
            J = 0
            for k in range(N):
                dx = x[k] - xPrev[k]
                if init:
                    du = np.zeros(m)
                else:
                    du = alpha * lArr[k] + LArr[k] @ dx
                u[k] = u[k] + du
                J += self.c(k, x[k], u[k])
                x[k + 1] = self.f(k, x[k], u[k])
            J += self.cf(x[-1])

            if (J - J_prev) / dJ_exp > self.cLineSearch:
                converged = True
                break

            alpha = alpha * self.betaLineSearch

        if not converged:
            raise RuntimeError("lineSearch failed to converge!!!")

        return x, u, J

    def _backwardPass(self, x, u, mu, delta):
        N, m = u.shape
        n = x.shape[1]

        cf = self.cf(x[N])
        cfx = self.cfx(x[N])
        cfxx = self.cfxx(x[N])

        lArr = np.zeros((N, m))
        LArr = np.zeros((N, m, n))

        maxRegularizationIter = 100
        for regIter in range(maxRegularizationIter):
            v = cf
            vVec = cfx
            V = cfxx
            dJ_lin = 0
            dJ_quad = 0
            converged = True
            for k in range(N - 1, -1, -1):
                Q, Qx, Qu, Qxx, Quu, Qux = self._computeQ(k, x[k], u[k], V, vVec, v, mu)

                if not (np.all(np.linalg.eigvals(Quu) > 0)):
                    delta = max(self.delta0, delta * self.delta0)
                    mu = max(self.muMin, mu * delta)
                    converged = False
                    break

                l = -np.linalg.solve(Quu, Qu)
                L = -np.linalg.solve(Quu, Qux)

                dv_lin = l.T @ Qu
                dv_quad = 0.5 * l.T @ Quu @ l

                v = dv_lin + dv_quad
                vVec = Qx + L.T @ Qu + Qux.T @ l + L.T @ Quu @ l
                V = Qxx + L.T @ Qux + Qux.T @ L + L.T @ Quu @ L

                lArr[k] = l
                LArr[k] = L

                dJ_lin += dv_lin
                dJ_quad += dv_quad

            if converged:
                delta = min(1 / self.delta0, delta / self.delta0)
                mu1 = mu * delta
                mu = mu1 if mu1 > self.muMin else 0
                break

        return LArr, lArr, mu, delta, dJ_lin, dJ_quad


class DDP(iLQR):

    def __init__(
        self,
        dynFun: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
        costFun: Callable[[int, np.ndarray, np.ndarray], float],
        x0: np.ndarray,
        u: np.ndarray,
        terminalCostFun: Callable[[np.ndarray], float] = None,
        muMin: float = 1e-6,
        delta0: float = 2,
        maxIter: int = 100,
        tol: float = 1e-6,
        maxLineSearchIter: int = 16,
        betaLineSearch: float = 0.5,
        cLineSearch: float = 0.8,
        jittable: bool = True
    ):
        """
        Differential dynamic programming constructor

        Arguments
        ---------
        dynFun : Callable[[int, np.ndarray, np.ndarray], np.ndarray]
            Discrete dynamics function of the form `xOut = f(k,x,u)`
        costFun : Callable[[int, np.ndarray, np.ndarray], float]
            Discrete cost function of the form `j = c(k,x,u)`
        x0 : np.ndarray
            Initial state, array of shape (n,)
        u : np.ndarray,
            Initial guess for control trajectory, array of shape (N,m)
        terminalCostFun : Callable[[np.ndarray], float], optional
            Terminal cost function of the form: `j = cf(x)`
            Default: cf(x) = c(N,x,0)
        muMin : float
            Minimum cost regularization term, larger mu biases each step towards the current trajectory
            Default: 1e-6
        delta0 : float
            Minimal modification factor
            Default: 2
        maxIter : int, optional
            Maximum number of optimization iterations
            Default: 100
        tol : float, optional
            Convergence tolerance, will exit if norm(xPrev-x) <= tol
            Default: 1e-3
        maxLineSearchIter : int, optional
            Maximum number of line search iterations in forward pass
            Default: 16
        betaLineSearch : float, optional
            Scale factor per iteration of line search.
            Must be between 0 and 1
            Default: 0.5
        cLineSearch : float, optional
            Line search cost degredation threshold
            Must be between 0 and 1
            Default: 0.8
        jittable : bool, optional
            Specifies whether the dynamics and cost functions are compatible with jax.jit
            Default: True

        Notes
        -----
        Different dynamic programming is very dependent on the initial guess.
        If the algorithm is failing to converge, consider providing a different starting point.
        """
        super().__init__(
            dynFun,
            costFun,
            x0,
            u,
            terminalCostFun=terminalCostFun,
            muMin=muMin,
            delta0=delta0,
            maxIter=maxIter,
            tol=tol,
            maxLineSearchIter=maxLineSearchIter,
            betaLineSearch=betaLineSearch,
            cLineSearch=cLineSearch,
            jittable=jittable
        )

        # Compute additional derivatives
        fxx = jax.jacfwd(self.fx, 1)
        fux = jax.jacfwd(self.fu, 1)
        fuu = jax.jacfwd(self.fu, 2)
        if self.jittable:
            fxx = jax.jit(fxx)
            fux = jax.jit(fux)
            fuu = jax.jit(fuu)
        self.fxx = fxx
        self.fux = fux
        self.fuu = fuu

    def _computeQ(self, k, x, u, V, vVec, v, mu):
        """Note: we diverge from the TET paper here. The modified regularization doesn't seem to work well with DDP"""
        m = len(u)
        fxk = self.fx(k, x, u)
        fuk = self.fu(k, x, u)

        Q = self.c(k, x, u) + v
        Qx = self.cx(k, x, u) + fxk.T @ vVec
        Qu = self.cu(k, x, u) + fuk.T @ vVec
        Qxx = self.cxx(k, x, u) + fxk.T @ V @ fxk + np.einsum('i,ijk', vVec, self.fxx(k, x, u))
        Quu = self.cuu(k, x, u) + fuk.T @ V @ fuk + np.einsum('i,ijk', vVec, self.fuu(k, x, u))
        Qux = self.cux(k, x, u) + fuk.T @ V @ fxk + np.einsum('i,ijk', vVec, self.fux(k, x, u))
        Quu = Quu + mu * np.eye(m)
        return Q, Qx, Qu, Qxx, Quu, Qux
