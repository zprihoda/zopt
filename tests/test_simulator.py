import numpy as np

from zProj.simulator import Simulator, SimBlock


def dynFun(t, x, u):
    return np.array([]), -x + u


def controlFun(t, xCtrl, xDyn):
    dxCtrl = np.array([])
    u = 0.1 * xDyn
    return u, dxCtrl


def test_simBlockContinuous():
    SimBlock(dynFun, np.array([0]), dt=0, jittable=False, name="dynamics")


def test_simluatorInitContinuous():
    controlBlock = SimBlock(controlFun, np.array([]), dt=0, jittable=False, name="controller")
    dynBlock = SimBlock(dynFun, np.array([0.]), dt=0, jittable=False, name="dynamics")
    Simulator([controlBlock, dynBlock], (0, 1))


def test_simulateContinuous():
    controlBlock = SimBlock(controlFun, np.array([]), dt=0, jittable=False, name="controller")
    dynBlock = SimBlock(dynFun, np.array([0.]), dt=0, jittable=False, name="dynamics")
    sim = Simulator([controlBlock, dynBlock], (0, 1))
    tArr, xCtrlArr, xDynArr, yCtrlArr, yDynArr = sim.simulate()
    assert len(tArr) == xDynArr.shape[0]
    assert len(tArr) == xCtrlArr.shape[0]
    assert len(tArr) == yCtrlArr.shape[0]
    assert len(tArr) == yDynArr.shape[0]


def test_simBlockDiscrete():
    SimBlock(dynFun, np.array([0]), dt=1, jittable=False, name="dynamics")


def test_simluatorInitDiscrete():
    controlBlock = SimBlock(controlFun, np.array([]), dt=1, jittable=False, name="controller")
    dynBlock = SimBlock(dynFun, np.array([0.]), dt=1, jittable=False, name="dynamics")
    Simulator([controlBlock, dynBlock], (0, 1))


def test_simulateDiscrete():
    controlBlock = SimBlock(controlFun, np.array([]), dt=0.1, jittable=False, name="controller")
    dynBlock = SimBlock(dynFun, np.array([0.]), dt=0.1, jittable=False, name="dynamics")
    sim = Simulator([controlBlock, dynBlock], (0, 0.2))
    tArr, xCtrlArr, xDynArr, yCtrlArr, yDynArr = sim.simulate()
    assert len(tArr) == xDynArr.shape[0]
    assert len(tArr) == xCtrlArr.shape[0]
    assert len(tArr) == yCtrlArr.shape[0] + 1  # For discrete sims, we return up to t[N], x[N], y[N-1]
    assert len(tArr) == yDynArr.shape[0] + 1
