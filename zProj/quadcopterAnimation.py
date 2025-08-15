import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from zProj.quadcopter import Quadcopter


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

    def _getRectangularPrism(self, center, dx, dy, dz, R=np.eye(3), **kwargs):
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
        prism = Poly3DCollection(faces, **kwargs)
        return prism

    def _plotCylinder(self, ax, center, r, dz, R=np.eye(3), N=100, **kwargs):
        theta = np.linspace(0, 2 * np.pi, N)
        z = np.array([-dz / 2, dz / 2])
        theta, z = np.meshgrid(theta, z)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = (center[None, :, None] + (R @ np.stack([x, y, z]).transpose(2, 0, 1))).transpose(1, 2, 0)
        ax.plot_surface(points[0], points[1], points[2], **kwargs)

        # TODO: Add top and bottom rotor surface?
        return None

    def _getBody(self, x, R):
        center = x[9:12]
        body = self._getRectangularPrism(
            center,
            self.bodyWidth,
            self.bodyWidth,
            self.bodyHeight,
            R=R,
            facecolors="cyan",
            linewidths=1,
            edgecolors='k',
            zorder=1
        )
        return body

    def _getArms(self, x, R):
        th = np.pi / 4
        R_body2arm = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        R_arm = R @ R_body2arm
        center_body = x[9:12]
        center1 = center_body + R_arm / 2 @ (self.armLength * np.array([1, 0, 0]))
        arm1 = self._getRectangularPrism(
            center1,
            self.armLength,
            self.armWidth,
            self.armWidth,
            R=R_arm,
            facecolors="cyan",
            linewidths=1,
            edgecolors='k'
        )
        center2 = center_body + R_arm / 2 @ (self.armLength * np.array([-1, 0, 0]))
        arm2 = self._getRectangularPrism(
            center2,
            self.armLength,
            self.armWidth,
            self.armWidth,
            R=R_arm,
            facecolors="cyan",
            linewidths=1,
            edgecolors='k'
        )
        center3 = center_body + R_arm / 2 @ (self.armLength * np.array([0, 1, 0]))
        arm3 = self._getRectangularPrism(
            center3,
            self.armWidth,
            self.armLength,
            self.armWidth,
            R=R_arm,
            facecolors="cyan",
            linewidths=1,
            edgecolors='k'
        )
        center4 = center_body + R_arm / 2 @ (self.armLength * np.array([0, -1, 0]))
        arm4 = self._getRectangularPrism(
            center4,
            self.armWidth,
            self.armLength,
            self.armWidth,
            R=R_arm,
            facecolors="cyan",
            linewidths=1,
            edgecolors='k'
        )
        return [arm1, arm2, arm3, arm4]

    def _addRotors(self, ax, x, R):
        center_body = x[9:12]
        center1 = center_body + R @ (
            self.armLength * np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0]) -
            np.array([0, 0, self.armWidth / 2 + self.rotorHeight / 2])
        )
        center2 = center_body + R @ (
            self.armLength * np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0]) -
            np.array([0, 0, self.armWidth / 2 + self.rotorHeight / 2])
        )
        center3 = center_body + R @ (
            self.armLength * np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0]) -
            np.array([0, 0, self.armWidth / 2 + self.rotorHeight / 2])
        )
        center4 = center_body + R @ (
            self.armLength * np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0]) -
            np.array([0, 0, self.armWidth / 2 + self.rotorHeight / 2])
        )
        self._plotCylinder(
            ax, center1, self.rotorRadius, self.rotorHeight, R=R, color="red", linewidths=1, edgecolors='k'
        )
        self._plotCylinder(
            ax, center2, self.rotorRadius, self.rotorHeight, R=R, color="red", linewidths=1, edgecolors='k'
        )
        self._plotCylinder(
            ax, center3, self.rotorRadius, self.rotorHeight, R=R, color="red", linewidths=1, edgecolors='k'
        )
        self._plotCylinder(
            ax, center4, self.rotorRadius, self.rotorHeight, R=R, color="red", linewidths=1, edgecolors='k'
        )

    def _initializePlot(self, x0: np.ndarray):
        x0[11] = -x0[11]  # Convert from down position to altitude
        phi, theta, psi = x0[6:9]
        ac = Quadcopter()
        R = np.array(ac._bodyToInertialRotationMatrix(phi, theta, psi))
        R[2, :] = -R[2, :]  # Reflect from NED to NEU

        body = self._getBody(x0, R)
        arms = self._getArms(x0, R)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection(arms[0])
        ax.add_collection(arms[1])
        ax.add_collection(arms[2])
        ax.add_collection(arms[3])
        ax.add_collection(body)
        self._addRotors(ax, x0, R)
        return fig, ax

    def _updatePlot(self, k, objs):
        pass

    def animate(self):
        fig = plt.figure()
        FuncAnimation(fig, self.updatePlot, range(self.N), self.initializePlot)


def main():
    t = [0, 1]
    x = np.zeros((2, 12))
    x[0, 9:12] = np.array([0, 0, 0])
    x[0, 6:9] = np.array([np.deg2rad(30), 0, 0])
    anim = QuadcopterAnimation(t, x)
    fig, ax = anim._initializePlot(x[0])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("N (m)")
    ax.set_ylabel("E (m)")
    ax.set_zlabel("h (m)")
    ax.view_init(azim=0, elev=0)
    plt.show()


if __name__ == "__main__":
    main()
