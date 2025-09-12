import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from zopt.quadcopter import Quadcopter
from zopt.lqrUtils import bilinearAffineLqr
from zopt.simulator import Simulator, SimBlock
from zopt.plottingTools import plotTimeTrajectory

rng = np.random.default_rng()


def controller(x, x0, u0, L, l):
    control = -L @ (x - x0) + u0 - l
    dxCtrl = np.array([])  # no controller states
    return control, dxCtrl


def main():
    # User inputs
    dt = 0.1
    N = 100
    x0Dyn = jnp.array([0., 0, 0, 0.5, 0.5, 0.1, 0, 0, 0, 0, 0, 0])

    # Get Linear dynamics
    ac = Quadcopter()
    x0, u0 = ac.trim(np.zeros(3))
    A, B = ac.linearize(x0, u0, dt=0.1)
    n, m = B.shape

    # Generate random problem
    A = jnp.repeat(A[None, :, :], N, axis=0)
    B = jnp.repeat(B[None, :, :], N, axis=0)
    d = jnp.zeros((N, n))
    Q = jnp.repeat(np.eye(n)[None, :, :], N, axis=0)
    R = jnp.repeat(np.eye(m)[None, :, :], N, axis=0)
    H = jnp.array(0.2 * rng.normal(size=(N, m, n)))  # random cross penalty on state-control
    qVec = 0.1 * jnp.repeat(np.array([1, -1, 0, 0, 0, 0, 0, 0])[None, :], N, axis=0)
    rVec = jnp.zeros((N, m))
    q = jnp.zeros(N)

    # Solve for finite horizon LQR
    L, l = bilinearAffineLqr(A, B, d, Q, R, H, qVec, rVec, q, N)

    # Simulate Problem
    dynamics = SimBlock(lambda k, x, u: (None, x + dt * ac.inertialDynamics(x, u)), x0Dyn, dt=dt, name="Dynamics")
    controllerBlock = SimBlock(
        lambda k, xCtrl, x: controller(x[:8], x0, u0, L[k], l[k]),
        np.array([]),
        dt=dt,
        name="Controller",
        jittable=False
    )
    t_span = (0, N * dt)
    sim = Simulator([controllerBlock, dynamics], t_span)
    tArr, _, xArr, uArr, _ = sim.simulate()

    # Plot Results
    plotTimeTrajectory(tArr, xArr[:, 0:3], names=['u', 'v', 'w'], title="Body Velocities")
    plotTimeTrajectory(tArr, xArr[:, 3:6], names=['p', 'q', 'r'], title="Body Rates")
    plotTimeTrajectory(tArr, xArr[:, 6:9], names=['phi', 'theta', 'psi'], title="Euler Angles")
    plotTimeTrajectory(tArr, xArr[:, 9:12], names=['x', 'y', 'z'], title="Positions")
    plotTimeTrajectory(tArr[:-1], uArr, names=["thrust", "pDot", "qDot", "rDot"], title="Pseudo Controls")
    plt.show()


if __name__ == "__main__":
    main()
