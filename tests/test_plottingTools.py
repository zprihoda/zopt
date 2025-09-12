import numpy as np
import matplotlib.pyplot as plt

from zopt.plottingTools import plotTimeTrajectory


def test_plotTimeTrajectory():
    tArr = np.array([0, 1])
    xArr = np.zeros((2, 2))
    names = ["x1", "x2"]
    fig = plotTimeTrajectory(tArr, xArr, names=names)
    plt.close(fig)


def test_plotTimeTrajectory_1dCase():
    tArr = np.array([0, 1])
    xArr = np.zeros((2, 1))
    names = ["x1"]
    fig = plotTimeTrajectory(tArr, xArr, names=names)
    plt.close(fig)


def test_plotTimeTrajectory_existingFig():
    tArr = np.array([0, 1])
    xArr = np.zeros((2, 2))
    names = ["x1", "x2"]
    fig, _ = plt.subplots(2, 1, squeeze=False)
    fig = plotTimeTrajectory(tArr, xArr, names=names, fig=fig)
    plt.close(fig)
