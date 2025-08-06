import jax.numpy as jnp
import pytest
import zProj.jaxUtils as jaxUtils


def test_interpMapped():
    x = jnp.array([0, 1])
    y = jnp.array([[0,2],
                   [1,3]])

    # Test interpolation
    assert jaxUtils.interpMapped(0, x, y) == pytest.approx(jnp.array([0,1]))
    assert jaxUtils.interpMapped(0.5, x, y) == pytest.approx(jnp.array([1,2]))
    assert jaxUtils.interpMapped(1, x, y) == pytest.approx(jnp.array([2,3]))

    # Test extrapolation (clipping)
    assert jaxUtils.interpMapped(-1, x, y) == pytest.approx(jnp.array([0,1]))
    assert jaxUtils.interpMapped(2, x, y) == pytest.approx(jnp.array([2,3]))
