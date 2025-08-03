import numpy as np

from zProj.plottingTools import plotTimeTrajectory


def test_plotTimeTrajectories():
    tArr = np.array([0, 1])
    xArr = np.zeros((2, 3))
    names = ["x1", "x2", "x3"]
    plotTimeTrajectory(tArr, xArr, names)


def test_plotTimeTrajectories_1dCase():
    tArr = np.array([0, 1])
    xArr = np.zeros((2, 1))
    names = ["x1"]
    plotTimeTrajectory(tArr, xArr, names)
