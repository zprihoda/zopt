import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from zProj.quadcopter import Quadcopter


def getRectangularPrismVertices(center: np.ndarray, dx: float, dy: float, dz: float, R: np.ndarray = np.eye(3)):
    """Get vertices for a rectangular prism"""
    vertex_pattern = np.array(list(product([-1, 1], repeat=3)))
    dv_body = (0.5 * np.array([dx, dy, dz])[None, :] * vertex_pattern)
    v = center + np.squeeze(R @ dv_body[:, :, None])
    faces = [
        [v[0], v[1], v[3], v[2]],
        [v[4], v[5], v[7], v[6]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[0], v[2], v[6], v[4]],
        [v[1], v[3], v[7], v[5]],
    ]
    return faces


class QuadcopterAnimation():

    def __init__(self, tTraj: np.ndarray, xTraj: np.ndarray):
        """
        Animate a quadcopter trajectory

        Arguments
        ---------
        tTraj : np.ndarray
            Array of time points; shape (N,)
        xTraj : np.ndarray
            Array of inertial states; shape (N, 12)
        """
        self.tTraj = tTraj
        self.xTraj = xTraj
        self.N = len(self.tTraj)
        self.bodyWidth = 0.1
        self.bodyHeight = 0.05
        self.armLength = 0.25
        self.armWidth = 0.02
        self.rotorRadius = 0.05
        self.rotorHeight = 0.01

        self.ac = Quadcopter()  # For body to inertial rotation matrix

    def _plotCylinder(self, ax, center, r, dz, R=np.eye(3), N=50, **kwargs):
        theta = np.linspace(0, 2 * np.pi, N)
        z = np.array([-dz / 2, dz / 2])
        theta, z = np.meshgrid(theta, z)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = (center[None, :, None] + (R @ np.stack([x, y, z]).transpose(2, 0, 1))).transpose(1, 2, 0)
        ax.plot_surface(points[0], points[1], points[2], **kwargs)

        # TODO: Add top and bottom rotor surface?
        return None

    def _getBodyVerts(self, x, R_body2enu, R_ned2enu):
        center = R_ned2enu @ x[9:12]
        faces = getRectangularPrismVertices(center, self.bodyWidth, self.bodyWidth, self.bodyHeight, R=R_body2enu)
        return faces

    def _getArmVerts(self, x, R_body2enu, R_ned2enu):
        th = np.pi / 4
        R_arm2body = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        R_arm = R_body2enu @ R_arm2body
        center_enu = R_ned2enu @ x[9:12]
        l = self.armLength
        w = self.armWidth

        center1 = center_enu + R_arm @ (0.5 * self.armLength * np.array([1, 0, 0]))
        verts1 = getRectangularPrismVertices(center1, l, w, w, R=R_arm)

        center2 = center_enu + R_arm @ (0.5 * self.armLength * np.array([-1, 0, 0]))
        verts2 = getRectangularPrismVertices(center2, l, w, w, R=R_arm)

        center3 = center_enu + R_arm @ (0.5 * self.armLength * np.array([0, 1, 0]))
        verts3 = getRectangularPrismVertices(center3, w, l, w, R=R_arm)

        center4 = center_enu + R_arm @ (0.5 * self.armLength * np.array([0, -1, 0]))
        verts4 = getRectangularPrismVertices(center4, w, l, w, R=R_arm)
        return [verts1, verts2, verts3, verts4]

    def _getHeadingVecPoints(self, x0, R_body2enu, R_ned2enu):
        pos_ned = x0[9:12]
        pos_enu = R_ned2enu @ pos_ned
        x_body = 2 * self.bodyWidth / 2 * np.array([1, 0, 0])
        x_enu = R_body2enu @ x_body
        vecStart = pos_enu + R_body2enu @ np.array([0, 0, -self.bodyHeight / 2])
        vecEnd = vecStart + x_enu
        return np.stack([vecStart, vecEnd], axis=1)

    def _addRotors(self, ax, x, R_body2enu, R_ned2enu):
        center_enu = R_ned2enu @ x[9:12]
        center1 = center_enu + R_body2enu @ (
            self.armLength * np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0]) -
            np.array([0, 0, self.armWidth / 2 + self.rotorHeight / 2])
        )
        center2 = center_enu + R_body2enu @ (
            self.armLength * np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0]) -
            np.array([0, 0, self.armWidth / 2 + self.rotorHeight / 2])
        )
        center3 = center_enu + R_body2enu @ (
            self.armLength * np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0]) -
            np.array([0, 0, self.armWidth / 2 + self.rotorHeight / 2])
        )
        center4 = center_enu + R_body2enu @ (
            self.armLength * np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0]) -
            np.array([0, 0, self.armWidth / 2 + self.rotorHeight / 2])
        )
        self._plotCylinder(
            ax, center1, self.rotorRadius, self.rotorHeight, R=R_body2enu, color="red", linewidths=1, edgecolors='k'
        )
        self._plotCylinder(
            ax, center2, self.rotorRadius, self.rotorHeight, R=R_body2enu, color="red", linewidths=1, edgecolors='k'
        )
        self._plotCylinder(
            ax, center3, self.rotorRadius, self.rotorHeight, R=R_body2enu, color="red", linewidths=1, edgecolors='k'
        )
        self._plotCylinder(
            ax, center4, self.rotorRadius, self.rotorHeight, R=R_body2enu, color="red", linewidths=1, edgecolors='k'
        )

    def _initializePlot(self, x0: np.ndarray):

        # Get body to ENU rotation matrix
        phi, theta, psi = x0[6:9]
        R_body2ned = np.array(self.ac._bodyToInertialRotationMatrix(phi, theta, psi))
        R_ned2enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        R_body2enu = R_ned2enu @ R_body2ned

        # Initialize objects
        verts = self._getBodyVerts(x0, R_body2enu, R_ned2enu)
        body = Poly3DCollection(verts, facecolors="cyan", linewidths=1, edgecolors='k')

        verts = self._getArmVerts(x0, R_body2enu, R_ned2enu)
        arms = [Poly3DCollection(verts[i], facecolors="cyan", linewidths=1, edgecolors='k', zorder=1) for i in range(4)]

        vec = self._getHeadingVecPoints(x0, R_body2enu, R_ned2enu)

        # Initialize figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection(arms[0])
        ax.add_collection(arms[1])
        ax.add_collection(arms[2])
        ax.add_collection(arms[3])
        ax.add_collection(body)
        self._addRotors(ax, x0, R_body2enu, R_ned2enu)  # TODO: Use Poly3DCollection for rotors to be compatible with funcAnimation
        headingLine = ax.plot(vec[0], vec[1], vec[2], "r-")[0]
        return fig, ax, (body, arms, headingLine)

    def _updatePlot(self, k, objs):
        pass

    def animate(self):
        fig = plt.figure()
        FuncAnimation(fig, self.updatePlot, range(self.N), self.initializePlot)


def main():
    t = [0, 1]
    x = np.zeros((2, 12))
    x[0, 9:12] = np.array([0, 0.5, 0])  # Position NED
    x[0, 6:9] = np.array([0, np.deg2rad(30), 0])  # phi,theta,psi
    anim = QuadcopterAnimation(t, x)
    fig, ax, objs = anim._initializePlot(x[0])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.set_zlabel("U (m)")
    plt.show()


if __name__ == "__main__":
    main()
