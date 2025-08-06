import numpy as np

from zProj.simulator import Simulator, SimBlock


def dynFun(t, x, u):
    return np.array([]), -x + u


def controlFun(t, xCtrl, xDyn):
    dxCtrl = np.array([])
    u = 0.1 * xDyn
    return u, dxCtrl


def test_simBlockInit():
    SimBlock(dynFun, np.array([0]), dt=0, jittable=False, name="dynamics")


def test_simluatorInit():
    controlBlock = SimBlock(controlFun, np.array([]), dt=0, jittable=False, name="controller")
    dynBlock = SimBlock(dynFun, np.array([0.]), dt=0, jittable=False, name="dynamics")
    Simulator([controlBlock, dynBlock], (0, 1))


def test_simulate():
    controlBlock = SimBlock(controlFun, np.array([]), dt=0, jittable=False, name="controller")
    dynBlock = SimBlock(dynFun, np.array([0.]), dt=0, jittable=False, name="dynamics")
    sim = Simulator([controlBlock, dynBlock], (0, 1))
    tArr, xCtrlArr, xDynArr, yCtrlArr, yDynArr = sim.simulate()
    assert len(tArr) == xDynArr.shape[0]
    assert len(tArr) == xCtrlArr.shape[0]
    assert len(tArr) == yCtrlArr.shape[0]
    assert len(tArr) == yDynArr.shape[0]
