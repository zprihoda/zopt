import numpy as np
import pytest
import zProj.quadcopterAnimation as quadAni


def test_getRectangularPrismVertices():
    center = np.array([1, 2, 3])
    dx = 2
    dy = 4
    dz = 6
    verts = quadAni.getRectangularPrismVertices(center, dx, dy, dz)
    verts_arr = np.array(verts)
    assert np.all(np.max(verts_arr, axis=(0, 1)) == [2, 4, 6])
    assert np.all(np.min(verts_arr, axis=(0, 1)) == [0, 0, 0])


def test_getCylinderVertices():
    center = np.array([0, 0, 0])
    r = 1
    dz = 2
    verts = quadAni.getCylinderVertices(center, r, dz, N=5)
    verts_arr = np.array(verts)
    assert np.all(np.max(verts_arr, axis=(0, 1)) == pytest.approx([1, 1, 1]))
    assert np.all(np.min(verts_arr, axis=(0, 1)) == pytest.approx([-1, -1, -1]))


def test_QuadcopterAnimation_init():
    tTraj = np.array([0, 1])
    xTraj = np.zeros((2, 12))
    quadAni.QuadcopterAnimation(tTraj, xTraj)


@pytest.mark.filterwarnings("ignore:Animation was deleted")
def test_QuadcopterAnimation_animate():
    tTraj = np.array([0, 1])
    xTraj = np.zeros((2, 12))
    animObj = quadAni.QuadcopterAnimation(tTraj, xTraj)
    animObj.animate()
