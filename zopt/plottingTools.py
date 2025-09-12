import matplotlib.pyplot as plt
import numpy as np


def plotTimeTrajectory(tArr, xArr, names=None, title=None, fig=None):
    """
    Plot an array of time trajectories in separate subplots

    Arguments
    --------
        tArr: Time array of size (N,)
        xArr: State array of size (N,nx)
        names: Name of each state for ylabel
        title: title for plot
    """
    nx = xArr.shape[1]
    if names is None:
        names = ["x{i}" for i in range(nx)]

    if fig is None:
        newFig = True
        fig, axs = plt.subplots(nx, 1, sharex=True, squeeze=False)
    else:
        newFig = False
        axs = np.array(fig.axes).reshape((nx, 1))

    for ii in range(nx):
        axs[ii, 0].plot(tArr, xArr[:, ii])

        if newFig:
            axs[ii, 0].set_ylabel(names[ii])
            axs[ii, 0].grid()

    if newFig:
        axs[nx - 1, 0].set_xlabel("time (s)")

    if newFig and title is not None:
        axs[0, 0].set_title(title)

    return fig
