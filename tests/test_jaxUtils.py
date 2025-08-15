import jax.numpy as jnp
import pytest
import zProj.jaxUtils as jaxUtils

from unittest import mock


def test_interpMapped():
    x = jnp.array([0, 1])
    y = jnp.array([[0, 2], [1, 3]])

    # Test interpolation
    assert jaxUtils.interpMapped(0, x, y) == pytest.approx(jnp.array([0, 1]))
    assert jaxUtils.interpMapped(0.5, x, y) == pytest.approx(jnp.array([1, 2]))
    assert jaxUtils.interpMapped(1, x, y) == pytest.approx(jnp.array([2, 3]))

    # Test extrapolation (clipping)
    assert jaxUtils.interpMapped(-1, x, y) == pytest.approx(jnp.array([0, 1]))
    assert jaxUtils.interpMapped(2, x, y) == pytest.approx(jnp.array([2, 3]))


def test_maybeJit():
    func = lambda x: x**2
    with mock.patch("jax.jit") as jit_mock:
        jit_mock.return_value = lambda x: func(x)

        out = jaxUtils.maybeJit(func, False)
        jit_mock.assert_not_called()
        assert out(2) == 4

        out = jaxUtils.maybeJit(func, True)
        jit_mock.assert_called_once()
        assert out(3) == 9


class Foo:
    jittable = False

    @jaxUtils.maybeJitCls
    def func(self, x):
        return x**2


def test_maybeJitCls():
    foo = Foo()
    with mock.patch("jax.jit") as jit_mock:
        jit_mock.return_value = lambda _, x: x**2

        # False case
        assert foo.func(4) == 16
        jit_mock.assert_not_called()

        # True case
        foo.jittable = True
        assert foo.func(5) == 25
        jit_mock.assert_called_once()
