import numpy as np

from zProj.plottingTools import plotTimeTrajectory


def test_plotTimeTrajectory():
    tArr = np.array([0, 1])
    xArr = np.zeros((2, 2))
    names = ["x1", "x2"]
    plotTimeTrajectory(tArr, xArr, names)


def test_plotTimeTrajectory_1dCase():
    tArr = np.array([0, 1])
    xArr = np.zeros((2, 1))
    names = ["x1"]
    plotTimeTrajectory(tArr, xArr, names)
