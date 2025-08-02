import numpy as np

from zProj.plottingTools import plotTimeTrajectories


def test_plotTimeTrajectories():
    tArr = np.array([0, 1])
    xArr = np.zeros((2, 3))
    uArr = np.ones((2, 1))
    stateGroupNames = ["States 1", "State2"]
    stateGroups = [["x1", "x2"], ["x3"]]
    controlGroupNames = ["Controls"]
    controlGroups = [["u1"]]
    plotTimeTrajectories(tArr, xArr, uArr, stateGroupNames, stateGroups, controlGroupNames, controlGroups)
