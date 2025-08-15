import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as spo

from functools import partial
from itertools import product

jax.config.update("jax_enable_x64", True)  # Required for trim minimization to converge


class Quadcopter():
    """Quadcopter Object"""

    def __init__(self):
        """Initialize a quadcopter object"""
        self.g = 9.807  # gravity (m / s**2)
        self.m = 2.5  # mass (kg)
        self.I = jnp.eye(3)  # Inertia tensor
        self.I_inv = jnp.linalg.inv(self.I)

        self._linFunc = jax.jit(jax.jacobian(self.rigidBodyDynamics, argnums=(0, 1)))

    @partial(jax.jit, static_argnames=['self'])
    def _bodyToInertialRotationMatrix(self, phi: float, theta: float, psi: float) -> jnp.ndarray:
        """Comptue body-to-inertial rotation matrix"""
        cphi = jnp.cos(phi)
        sphi = jnp.sin(phi)
        cth = jnp.cos(theta)
        sth = jnp.sin(theta)
        cpsi = jnp.cos(psi)
        spsi = jnp.sin(psi)
        R = jnp.array(
            [
                [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi - sphi * spsi],
                [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
                [-sth, sphi * cth, cphi * cth],
            ]
        )
        return R

    @partial(jax.jit, static_argnames=['self'])
    def _bodyRatesToEulerRatesRotationMatrix(self, phi: float, theta: float) -> jnp.ndarray:
        """Compute body-angular-rates to euler-rates rotation matrix"""
        sphi = jnp.sin(phi)
        cphi = jnp.cos(phi)
        cth = jnp.cos(theta)
        tth = jnp.tan(theta)
        R_rates2Eul = jnp.array([[1, sphi * tth, cphi * tth], [0, cphi, -sphi], [0, sphi / cth, cphi / cth]])
        return R_rates2Eul

    @partial(jax.jit, static_argnames=['self'])
    def _getAeroForceMomemnts(
        self, state: jnp.ndarray, windBody: jnp.ndarray = jnp.zeros(3)
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the aero force/moments"""
        uvw = state[0:3]
        pqr = state[3:6]

        # ADB functions
        force_lin = jnp.array([-0.2, -0.2, -0.3])
        force_quad = jnp.array([-0.05, -0.05, -0.1])
        moment_lin = jnp.array([-0.1, -0.1, -0.05])

        # Compute aero force moments
        uvw_aero = uvw - windBody  # body-frame velocities wrt air
        force_aero = force_lin * uvw_aero + force_quad * uvw_aero**2
        moment_aero = moment_lin * pqr
        return force_aero, moment_aero

    @partial(jax.jit, static_argnames=['self'])
    def rigidBodyDynamics(
        self, state: jnp.ndarray, control: jnp.ndarray, wind_body: jnp.ndarray = jnp.zeros(3)
    ) -> jnp.ndarray:
        """
        Rigid-body dynamics function for quadcopter `xDot = f(x,u)`

        Arguments
        ---------
            state : aircraft state, [u,v,w,p,q,r,phi,theta]
            control : control input, [-fz,mx,my,mz]   (all mass/inertia normalized, accelerations)
            wind_body : wind in the body frame

        Returns
        -------
            dState : time derivative of state
        """
        # Unpack inputs
        uvw = state[0:3]
        pqr = state[3:6]
        phi, theta = state[6:8]
        thrust = control[0]  # vertical acceleration (thrust)
        mxyz = control[1:4]  # angular accelerations (moments)

        # Get rotation matrices
        d2xyz = jnp.array([-jnp.sin(theta), jnp.sin(phi) * jnp.cos(theta), jnp.cos(phi) * jnp.cos(theta)])
        R_rates2Eul = self._bodyRatesToEulerRatesRotationMatrix(phi, theta)

        # Get total force / moments in body axis
        force_aero, moment_aero = self._getAeroForceMomemnts(state, wind_body)

        force_control = self.m * jnp.array([0, 0, -thrust])
        force_gravity = self.m * self.g * d2xyz
        force_total = force_control + force_aero + force_gravity

        moment_control = self.I @ mxyz
        moment_total = moment_control + moment_aero

        # Equations of motion
        uvwDot = (1 / self.m) * (-jnp.cross(pqr, uvw) + force_total)
        pqrDot = self.I_inv @ (-jnp.cross(pqr, self.I @ pqr) + moment_total)
        phiThetaDot = R_rates2Eul[0:2, :] @ pqr

        dState = jnp.concatenate([uvwDot, pqrDot, phiThetaDot])
        return dState

    @partial(jax.jit, static_argnames=['self'])
    def inertialDynamics(
        self, state: jnp.ndarray, control: jnp.ndarray, wind_ned: jnp.ndarray = jnp.zeros(3)
    ) -> jnp.ndarray:
        """
        Dynamics function for quadcopter with position states `xDot = f(x,u)`

        Arguments
        ---------
            state : aircraft state, [u,v,w,p,q,r,phi,theta,psi,x,y,z]
            control : control input, [-fz,mx,my,mz]   (all mass/inertia normalized, accelerations)
            wind_ned : wind in the north-east-down frame

        Returns
        -------
            dState : time derivative of state
        """
        uvw = state[0:3]
        pqr = state[3:6]
        phi, theta, psi = state[6:9]
        R_b2i = self._bodyToInertialRotationMatrix(phi, theta, psi)
        R_rates2Eul = self._bodyRatesToEulerRatesRotationMatrix(phi, theta)

        wind_body = R_b2i.T @ wind_ned
        xDot_rb = self.rigidBodyDynamics(state[:9], control, wind_body=wind_body)

        psiDot = jnp.array([R_rates2Eul[2, :] @ pqr])
        xyzDot = R_b2i @ uvw
        xDot_inertial = jnp.concatenate([xDot_rb, psiDot, xyzDot])
        return xDot_inertial

    def trim(self, uvwTrim: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Trim quadcopter at specified uvw

        Arguments
        ---------
            uvwTrim: Vector specifying body velocities at which to trim

        Returns
        -------
            xTrim: Trim state vector
            uTrim: Trim control vector
        """
        nxz = 5

        def _getXu(z):
            x = jnp.concatenate([uvwTrim, z[:nxz]])
            u = z[nxz:]
            return x, u

        x0 = jnp.zeros(nxz)
        u0 = jnp.array([self.g, 0, 0, 0])
        z0 = jnp.concatenate([x0, u0])
        trimFunc = lambda z: jnp.sum(self.rigidBodyDynamics(*_getXu(z))**2)
        trimFunc = jax.jit(trimFunc)
        out = spo.minimize(trimFunc, z0, method="BFGS")

        if not out.success:
            raise RuntimeError("Trim failed")

        xTrim, uTrim = _getXu(out.x)
        return xTrim, uTrim

    def linearize(self, x0: jnp.ndarray, u0: jnp.ndarray, dt: float = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Linearize the quadcopter dynamics about a given state and control input

        Arguments
        ---------
            x0 : Rigid body state about which to linearize
            u0 : Control about which to linearize
            dt : Sample length for discrete linearization. Set dt=0 for continuous time linearization
                 Default value: 0 (continuous time)
        Returns
        -------
            A : Partial derivative of the dynamics function wrt the state input
            B : Partial derivative of the dynamics function wrt the control input
        """
        A, B = self._linFunc(x0, u0)

        if dt != 0:
            # Forward Euler Discretization
            A = jnp.eye(A.shape[0]) + dt * A
            B = dt * B

        return A, B


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class QuadcopterAnimation():
    def __init__(self, tTraj: jnp.ndarray, xTraj: jnp.ndarray):
        """
        Animate a quadcopter trajectory

        Arguments
        ---------
        tTraj : jnp.ndarray
            Array of time points; shape (N,)
        xTraj : jnp.ndarray
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

    def _getRectangularPrism(self, center, dx, dy, dz, R = jnp.eye(3), **kwargs):
        vertex_pattern = np.array(list(product([-1,1], repeat=3)))
        dv_body = (0.5 * np.array([dx,dy,dz])[None, :] * vertex_pattern)
        v = center + np.squeeze(R @ dv_body[:,:,None])
        faces = [
            [v[0], v[1], v[3], v[2]],
            [v[4], v[5], v[7], v[6]],
            [v[0], v[1], v[5], v[4]],
            [v[2], v[3], v[7], v[6]],
            [v[0], v[2], v[6], v[4]],
            [v[1], v[3], v[7], v[5]]
        ]
        prism = Poly3DCollection(faces, **kwargs)
        return prism

    def _plotCylinder(self, ax, center, r, dz, R=jnp.eye(3), N=100, **kwargs):
        theta = np.linspace(0, 2 * np.pi, N)
        z = np.array([center[2]-dz/2, center[2]+dz/2])
        theta, z = np.meshgrid(theta, z)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = (center[None,:,None] + (R @ np.stack([x,y,z]).transpose(2,0,1))).transpose(1,2,0)
        ax.plot_surface(points[0], points[1], points[2], **kwargs)
        # TODO: Add top and bottom rotor surface
        return None


    def _getBody(self, x, R):
        center = x[9:12]
        body = self._getRectangularPrism(center, self.bodyWidth, self.bodyWidth, self.bodyHeight, R=R, facecolors="cyan", linewidths=1, edgecolors='k', zorder=1)
        return body

    def _getArms(self, x, R):
        th = np.pi/4
        R_body2arm = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        R_arm = R_body2arm @ R
        center_body = x[9:12]
        center1 = center_body + R_arm/2 @ (self.armLength * jnp.array([ 1, 0, 0]))
        arm1 = self._getRectangularPrism(center1, self.armLength, self.armWidth, self.armWidth, R=R_arm, facecolors="cyan", linewidths=1, edgecolors='k')
        center2 = center_body + R_arm/2 @ (self.armLength * jnp.array([-1, 0, 0]))
        arm2 = self._getRectangularPrism(center2, self.armLength, self.armWidth, self.armWidth, R=R_arm, facecolors="cyan", linewidths=1, edgecolors='k')
        center3 = center_body + R_arm/2 @ (self.armLength * jnp.array([ 0, 1, 0]))
        arm3 = self._getRectangularPrism(center3, self.armWidth, self.armLength, self.armWidth, R=R_arm, facecolors="cyan", linewidths=1, edgecolors='k')
        center4 = center_body + R_arm/2 @ (self.armLength * jnp.array([ 0,-1, 0]))
        arm4 = self._getRectangularPrism(center4, self.armWidth, self.armLength, self.armWidth, R=R_arm, facecolors="cyan", linewidths=1, edgecolors='k')
        return [arm1, arm2, arm3, arm4]

    def _addRotors(self, ax, x, R):
        center_body = x[9:12]
        center1 = center_body + R @ (self.armLength*jnp.array([ 1/np.sqrt(2), 1/np.sqrt(2), 0]) + jnp.array([0, 0, self.armWidth/2+self.rotorHeight/2]))
        center2 = center_body + R @ (self.armLength*jnp.array([ 1/np.sqrt(2),-1/np.sqrt(2), 0]) + jnp.array([0, 0, self.armWidth/2+self.rotorHeight/2]))
        center3 = center_body + R @ (self.armLength*jnp.array([-1/np.sqrt(2),-1/np.sqrt(2), 0]) + jnp.array([0, 0, self.armWidth/2+self.rotorHeight/2]))
        center4 = center_body + R @ (self.armLength*jnp.array([-1/np.sqrt(2), 1/np.sqrt(2), 0]) + jnp.array([0, 0, self.armWidth/2+self.rotorHeight/2]))
        self._plotCylinder(ax, center1, self.rotorRadius, self.rotorHeight, R=R, color="red", linewidths=1, edgecolors='k')
        self._plotCylinder(ax, center2, self.rotorRadius, self.rotorHeight, R=R, color="red", linewidths=1, edgecolors='k')
        self._plotCylinder(ax, center3, self.rotorRadius, self.rotorHeight, R=R, color="red", linewidths=1, edgecolors='k')
        self._plotCylinder(ax, center4, self.rotorRadius, self.rotorHeight, R=R, color="red", linewidths=1, edgecolors='k')

    def _initializePlot(self, x0: jnp.ndarray):
        phi,theta,psi = x0[6:9]
        ac = Quadcopter()
        R = ac._bodyToInertialRotationMatrix(phi, theta, psi)

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
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        plt.show()
        return None


    def _updatePlot(self, k, objs):
        x = self.xTraj[k]

    def animate(self):
        fig = plt.figure()
        FuncAnimation(fig, self.updatePlot, range(self.N), self.initializePlot)

t = [0,1]
x = np.zeros((2,12))
anim = QuadcopterAnimation(t, x)
anim._initializePlot(x[0])
