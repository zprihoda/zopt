import matplotlib.pyplot as plt


def plotTimeTrajectory(tArr, xArr, names=None, title=None):
    """
    Plot an array of time trajectories in separate subplots

    Arguments
    --------
        tArr: Time array of size (N,)
        xArr: State array of size (N,nx)
        names: Name of each state for ylabel
        title: title for plot
    """

    idx = 0
    nx = xArr.shape[1]
    if names is None:
        names = ["x{i}" for i in range(nx)]

    fig, axs = plt.subplots(nx, 1, sharex=True, squeeze=False)
    for ii in range(nx):
        axs[ii, 0].plot(tArr, xArr[:, idx])
        idx += 1
        axs[ii, 0].set_ylabel(names[ii])
        axs[ii, 0].grid()
    axs[nx - 1, 0].set_xlabel("time (s)")

    if title is not None:
        axs[0, 0].set_title(title)
