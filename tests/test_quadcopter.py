import numpy as np
import pytest

from zProj.quadcopter import Quadcopter


def test_default_init():
    """Test the default class initialization runs without error"""
    Quadcopter()


def test_bodyToInertialRotationMatrix():
    ac = Quadcopter()
    assert ac._bodyToInertialRotationMatrix(0, 0, 0) == pytest.approx(np.eye(3))

    th = np.pi / 6
    cth = np.cos(th)
    sth = np.sin(th)

    R_exp = np.array([[1, 0, 0], [0, cth, -sth], [0, sth, cth]])
    assert ac._bodyToInertialRotationMatrix(th, 0, 0) == pytest.approx(R_exp)

    R_exp = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    assert ac._bodyToInertialRotationMatrix(0, th, 0) == pytest.approx(R_exp)

    R_exp = np.array([[cth, -sth, 0], [sth, cth, 0], [0, 0, 1]])
    assert ac._bodyToInertialRotationMatrix(0, 0, th) == pytest.approx(R_exp)


def test_bodyRatesToEulerRatesRotationMatrix():
    ac = Quadcopter()
    assert ac._bodyRatesToEulerRatesRotationMatrix(0, 0) == pytest.approx(np.eye(3))

    th = np.pi / 6
    cth = np.cos(th)
    sth = np.sin(th)
    tth = np.tan(th)

    R_exp = np.array([[1, 0, 0], [0, cth, -sth], [0, sth, cth]])
    assert ac._bodyRatesToEulerRatesRotationMatrix(th, 0) == pytest.approx(R_exp)

    R_exp = np.array([[1, 0, tth], [0, 1, 0], [0, 0, 1 / cth]])
    assert ac._bodyRatesToEulerRatesRotationMatrix(0, th) == pytest.approx(R_exp)


def test_rigidBodyDynamics():
    ac = Quadcopter()
    state = np.zeros(9)
    control = np.zeros(4)
    xDot = ac.rigidBodyDynamics(state, control)
    xDot_exp = np.array([0, 0, 9.807, 0, 0, 0, 0, 0, 0])
    assert xDot == pytest.approx(xDot_exp)

    control = np.array([9.807, 0, 0, 0])
    xDot = ac.rigidBodyDynamics(state, control)
    xDot_exp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert xDot == pytest.approx(xDot_exp)


def test_inertialDynamics():
    ac = Quadcopter()

    # No motion case
    state = np.zeros(12)
    control = np.array([9.807, 0, 0, 0])
    xDot = ac.inertialDynamics(state, control)
    xDot_exp = np.zeros(12)
    assert xDot == pytest.approx(xDot_exp)

    # No rotation case
    uvw = np.array([0.1, 0.2, 0.3])
    state = np.zeros(12)
    state[0:3] = uvw
    xDot = ac.inertialDynamics(state, control)
    xyzDot_exp = uvw
    assert xDot[9:] == pytest.approx(xyzDot_exp)

    # Rotation case
    uvw = np.array([0.1, 0.2, 0.3])
    psi = np.pi / 2
    state = np.zeros(12)
    state[0:3] = uvw
    state[8] = psi
    xDot = ac.inertialDynamics(state, control)
    xyzDot_exp = np.array([-uvw[1], uvw[0], uvw[2]])
    assert xDot[9:] == pytest.approx(xyzDot_exp)
