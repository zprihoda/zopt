import numpy as np

from zProj.simulator import Simulator


def dynFun(t, x, u):
    return -x + u


def controlFun(t, x):
    return 0.1 * x


def test_init():
    Simulator(dynFun, controlFun, (0, 1), np.array([0]))


def test_simulate():
    sim = Simulator(dynFun, controlFun, (0, 1), np.array([0]))
    tArr, xArr, uArr = sim.simulate()
    assert len(tArr) == xArr.shape[0]
    assert len(tArr) == uArr.shape[0]
