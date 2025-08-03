import jax
import jax.numpy as jnp
import scipy.optimize as spo

from functools import partial

jax.config.update("jax_enable_x64", True)


class Quadcopter():
    """Quadcopter Object"""

    def __init__(self):
        """Initialize a quadcopter object"""
        self.g = 9.807  # gravity (m / s**2)
        self.m = 2.5  # mass (kg)
        self.I = jnp.eye(3)  # Inertia tensor
        self.I_inv = jnp.linalg.inv(self.I)

        self._linFunc = jax.jit(jax.jacobian(self.rigidBodyDynamics, argnums=(0, 1)))

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

    def _bodyRatesToEulerRatesRotationMatrix(self, phi: float, theta: float) -> jnp.ndarray:
        """Compute body-angular-rates to euler-rates rotation matrix"""
        sphi = jnp.sin(phi)
        cphi = jnp.cos(phi)
        cth = jnp.cos(theta)
        tth = jnp.tan(theta)
        R_rates2Eul = jnp.array([[1, sphi * tth, cphi * tth], [0, cphi, -sphi], [0, sphi / cth, cphi / cth]])
        return R_rates2Eul

    def getAeroForceMomemnts(
        self, state: jnp.ndarray, windBody: jnp.ndarray = jnp.zeros(3)
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the aero force/moments"""
        uvw = state[0:3]
        pqr = state[3:6]

        # ADB functions
        force_lin = jnp.array([-0.1, -0.1, -0.2])
        force_quad = jnp.array([-0.05, -0.05, -0.1])
        moment_lin = jnp.array([-0.1, -0.1, -0.05])

        # Compute aero force moments
        uvw_aero = uvw - windBody  # body-frame velocities wrt air
        force_aero = force_lin * uvw_aero + force_quad * uvw_aero**2
        moment_aero = moment_lin * pqr
        return force_aero, moment_aero

    @partial(jax.jit, static_argnames=['self', 'returnRotMat'])
    def rigidBodyDynamics(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        wind_ned: jnp.ndarray = jnp.zeros(3),
        returnRotMat: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """
        Rigid-body dynamics function for quadcopter `xDot = f(x,u)`

        Arguments
        ---------
            state : aircraft state, [u,v,w,p,q,r,phi,theta,psi]
            control : control input, [-fz,mx,my,mz]   (all mass/inertia normalized, accelerations)
            wind_ned : wind in the north-east-down frame
            returnRotMat : Flag to return body to inertial rotation matrix
                Intended for use by inertial dynamics to avoid unnecessary recomputation

        Returns
        -------
            dState : time derivative of state
        """
        # Unpack inputs
        uvw = state[0:3]
        pqr = state[3:6]
        phi, theta, psi = state[6:9]
        thrust = control[0]  # vertical acceleration (thrust)
        mxyz = control[1:4]  # angular accelerations (moments)

        # Get rotation matrices
        R_b2i = self._bodyToInertialRotationMatrix(phi, theta, psi)
        R_rates2Eul = self._bodyRatesToEulerRatesRotationMatrix(phi, theta)

        # Get total force / moments in body axis
        wind_body = R_b2i.T @ wind_ned
        force_aero, moment_aero = self.getAeroForceMomemnts(state, wind_body)

        force_control = self.m * jnp.array([0, 0, -thrust])
        force_gravity = self.m * self.g * R_b2i[2, :]
        force_total = force_control + force_aero + force_gravity

        moment_control = self.I @ mxyz
        moment_total = moment_control + moment_aero

        # Equations of motion
        uvwDot = (1 / self.m) * (-jnp.cross(pqr, uvw) + force_total)
        pqrDot = self.I_inv @ (-jnp.cross(pqr, self.I @ pqr) + moment_total)
        eulDot = R_rates2Eul @ pqr

        dState = jnp.concatenate([uvwDot, pqrDot, eulDot])

        if returnRotMat:
            out = (dState, R_b2i)
        else:
            out = dState
        return out

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
        (xDot_rb, R_b2i) = self.rigidBodyDynamics(state[:9], control, wind_ned=wind_ned, returnRotMat=True)
        uvw = state[0:3]
        xyzDot = R_b2i @ uvw
        xDot_inertial = jnp.concatenate([xDot_rb, xyzDot])
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
        psiTrim = jnp.array([0])
        nxz = 5

        def _getXu(z):
            x = jnp.concatenate([uvwTrim, z[:nxz], psiTrim])
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

    def linearize(self, x0: jnp.ndarray, u0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Linearize the quadcopter dynamics about a given state and control input

        Arguments
        ---------
            x0 : State about which to linearize
            u0 : Control about which to linearize
        Returns
        -------
            A : Partial derivative of the dynamics function wrt the state input
            B : Partial derivative of the dynamics function wrt the control input
        """
        A, B = self._linFunc(x0, u0)
        return A, B
