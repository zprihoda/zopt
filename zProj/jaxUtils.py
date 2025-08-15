import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.jit, static_argnames=["left", "right", "period"])
@partial(jax.vmap, in_axes=(None, None, 0))
def interpMapped(x: jnp.ndarray, xp: jnp.ndarray, fp: float, left=None, right=None, period=None):
    """
    n-dimensional linear interpolation

    Arguments
    ---------
        x : N-dimensional array of x coordinates at which to evaluate the interpolation.
        xp : one-dimensional sorted array of points to be interpolated.
        fp : array of shape (n,xp.shape) containing the function values associated with xp.
        left, right, period : see jax interp documentation, defaults to clipped bounds

    Returns
    -------
        yq : (n,) array of interpolated vector values at xq
    """
    return jnp.interp(x, xp, fp, left=left, right=right, period=period)


def maybeJitCls(func):
    """Class method decorator: Jit function if self.jittable is true"""

    def _func(self, *args, **kwargs):
        if self.jittable:
            return jax.jit(func, static_argnames=['self'])(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return _func


def maybeJit(func, cond):
    """Jit a function if cond is true"""
    return jax.jit(func) if cond else func
