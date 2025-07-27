import numpy as np
import pytest

from zProj.quadcopter import Quadcopter

def test_default_init():
    """Test the default class initialization runs without error"""
    ac = Quadcopter()

def test_dynamics():
    ac = Quadcopter()
    state = np.zeros(12)
    control = np.zeros(4)
    xDot = ac.dynamics(state, control)
    xDot_exp = np.array([0,0,9.807,0,0,0,0,0,0,0,0,0])
    assert xDot == pytest.approx(xDot_exp)

    control = np.array([9.807,0,0,0])
    xDot = ac.dynamics(state, control)
    xDot_exp = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    assert xDot == pytest.approx(xDot_exp)
