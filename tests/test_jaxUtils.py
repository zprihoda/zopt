import jax.numpy as jnp
import pytest
import zProj.jaxUtils as jaxUtils


def test_interp1d():
    x = jnp.array([0, 1])
    y = jnp.array([[0, 2]])

    assert jaxUtils.interp1d(x, y, 0) == pytest.approx(jnp.array([0]))
    assert jaxUtils.interp1d(x, y, 0.5) == pytest.approx(jnp.array([1]))
    assert jaxUtils.interp1d(x, y, 1) == pytest.approx(jnp.array([2]))
