import numpy as np
import matplotlib.pyplot as plt

from zProj.lqrUtils import bilinearAffineLqr

# TODO: Consider converting this demo to use quadcopter and sim

def main():
    # User inputs
    n = 3
    m = 2
    N = 10

    # Generate random problem
    A = np.random.rand(N,n,n)
    B = np.random.rand(N,n,m)
    d = np.random.rand(N,n)
    Q = np.random.rand(N,n,n)
    R = np.random.rand(N,m,m)
    H = np.random.rand(N,m,n)
    qVec = np.random.rand(N,n)
    rVec = np.random.rand(N,m)
    q = np.random.rand(N)

    ## Enforce R > 0
    R = np.stack([R[i].T @ R[i] for i in range(N)], axis=0) + 0.1*np.eye(m)[None,:,:]

    # Solve for finite horizon LQR
    L,l = bilinearAffineLqr(A,B,d,Q,R,H,qVec,rVec,q,N)

    # Simulate Problem
    x = N*np.random.rand(n)
    xArr = np.zeros((N+1,n))
    xArr[0] = x
    uArr = np.zeros((N,m))
    for i in range(N):
        u = -L[i]@x - l[i]
        x = A[i]@x + B[i]@u + d[i]
        uArr[i] = u
        xArr[i+1] = x

    # Plot results
    fig,axs = plt.subplots(n,1,sharex=True)
    for i in range(n):
        axs[i].plot(range(N+1), xArr[:,i])
        axs[i].grid()
        axs[i].set_ylabel("$x_{:}$".format(i))
    axs[-1].set_xlabel("k")
    fig.suptitle("State Trajectory")

    fig,axs = plt.subplots(m,1,sharex=True)
    for i in range(m):
        axs[i].plot(range(N), uArr[:,i])
        axs[i].grid()
        axs[i].set_ylabel("$u_{:}$".format(i))
    axs[-1].set_xlabel("k")
    fig.suptitle("Control Trajectory")
    plt.show()

if __name__ == "__main__":
    main()
