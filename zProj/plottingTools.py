import matplotlib.pyplot as plt


def plotTimeTrajectories(tArr, xArr, uArr, stateGroupNames, stateGroups, controlGroupNames, controlGroups):

    # Plot State groups
    idx = 0
    for ii in range(len(stateGroupNames)):
        nx = len(stateGroups[ii])
        fig, axs = plt.subplots(nx, 1, sharex=True, squeeze=False)
        for jj in range(nx):
            axs[jj, 0].plot(tArr, xArr[:, idx])
            idx += 1
            axs[jj, 0].set_ylabel(stateGroups[ii][jj])
            axs[jj, 0].grid()
        axs[nx - 1, 0].set_xlabel("time (s)")
        fig.suptitle(stateGroupNames[ii])

    # Plot Control Groups
    idx = 0
    for ii in range(len(controlGroupNames)):
        nx = len(controlGroups[ii])
        fig, axs = plt.subplots(nx, 1, sharex=True, squeeze=False)
        for jj in range(nx):
            axs[jj, 0].plot(tArr, uArr[:, idx])
            idx += 1
            axs[jj, 0].set_ylabel(controlGroups[ii][jj])
            axs[jj, 0].grid()
        axs[nx - 1, 0].set_xlabel("time (s)")
        fig.suptitle(controlGroupNames[ii])
