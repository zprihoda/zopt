import jax
import jax.numpy as jnp


@jax.jit
def interp1d(x: jnp.ndarray, y: jnp.ndarray, xq: float):
    """
    Jax compliant linear vector interpolation

    Arguments
    ---------
        x : (N,) array of sorted sample points
        y : (n,N) array of vector values
        xq : x-value to evaluate interpolant at

    Returns
    -------
        yq : (n,) array of interpolated vector values at xq

    Notes
    -----
    - xq must be within the bounds [x[0], x[-1]]
    """
    idx = jnp.searchsorted(x, xq) - 1
    frac = (xq - x[idx]) / (x[idx + 1] - x[idx])
    yq = (1 - frac) * y[:, idx] + frac * y[:, idx + 1]
    return yq
