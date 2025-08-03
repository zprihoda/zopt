import numpy as np

from zProj.simulator import Simulator


def dynFun(t, x, u):
    return -x + u


def controlFun(t, xDyn, xCtrl):
    dxCtrl = np.array([])
    u = 0.1 * xDyn
    return dxCtrl, u


def test_init():
    Simulator(dynFun, controlFun, (0, 1), np.array([0]), np.array([]))


def test_simulate():
    sim = Simulator(dynFun, controlFun, (0, 1), np.array([0]), np.array([]))
    tArr, xDynArr, xCtrlArr, uArr = sim.simulate()
    assert len(tArr) == xDynArr.shape[0]
    assert len(tArr) == xCtrlArr.shape[0]
    assert len(tArr) == uArr.shape[0]
