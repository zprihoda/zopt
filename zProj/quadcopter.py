import jax as jax
import jax.numpy as np

class Quadcopter():
    """Quadcopter Object"""
    def __init__(self):
        """Initialize a quadcopter object"""
        self.g = 9.807  # gravity (m / s**2)
        self.m = 2.5    # mass (kg)
        self.I = np.eye(3)  # Inertia tensor
        self.I_inv = np.linalg.inv(self.I)

    def _bodyToInertialRotationMatrix(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """Comptue body-to-inertial rotation matrix"""
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        R = np.array([
            [cth*cpsi, cth*spsi, -sth],
            [sphi*sth*cpsi - cphi*spsi, sphi*sth*spsi + cphi*cpsi, sphi*cth],
            [cphi*sth*cpsi - sphi*spsi, cphi*sth*spsi - sphi*cpsi, cphi*cth]
        ])
        return R

    def _bodyRatesToEulerRatesRotationMatrix(self, phi: float, theta: float) -> np.ndarray:
        """Compute body-angular-rates to euler-rates rotation matrix"""
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        cth = np.cos(theta)
        tth = np.tan(theta)
        R_rates2Eul = np.array([
            [1, sphi*tth, cphi*tth],
            [0, cphi, -sphi],
            [0, sphi/cth, cphi/cth]
        ])
        return R_rates2Eul

    def getAeroForceMomemnts(self, state: np.ndarray, windBody: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the aero force/moments"""
        uvw = state[0:3]
        pqr = state[3:6]

        # ADB functions
        force_lin = np.array([-0.1, -0.1, -0.2])
        force_quad = np.array([-0.05, -0.05, -0.1])
        moment_lin = np.array([-0.1, -0.1, -0.05])

        # Compute aero force moments
        uvw_aero = uvw - windBody   # body-frame velocities wrt air
        force_aero = force_lin * uvw_aero +  force_quad * uvw_aero**2
        moment_aero = moment_lin * pqr
        return force_aero, moment_aero

    def dynamics(self, state: np.ndarray, control: np.ndarray, wind_ned: np.ndarray = np.zeros(3)):
        """
        Dynamics function for quadcopter object `xDot = f(x,u)`

        Arguments
        ---------
            state : aircraft state, [u,v,w,p,q,r,phi,theta,psi,x,y,z]
            control : control input, [-fz,mx,my,mz]   (all mass/inertia normalized, accelerations)
            wind_ned : wind in the north-east-down frame

        Returns
        -------
            dState : time derivative of state
        """
        # Unpack inputs
        uvw = state[0:3]
        pqr = state[3:6]
        phi,theta,psi = state[6:9]
        xyz = state[9:12]       # Inertial positions (ie. north, east, down)
        thrust = control[0]     # vertical acceleration (thrust)
        mxyz = control[1:4]     # angular accelerations (moments)

        # Get rotation matrices
        R_b2i = self._bodyToInertialRotationMatrix(phi, theta, psi)
        R_rates2Eul = self._bodyRatesToEulerRatesRotationMatrix(phi, theta)

        # Get total force / moments in body axis
        wind_body = R_b2i.T @ wind_ned
        force_aero,moment_aero = self.getAeroForceMomemnts(state, wind_body)

        force_control = self.m*np.array([0,0,-thrust])
        force_gravity = self.m*self.g*R_b2i[:,2]
        force_total = force_control + force_aero + force_gravity

        moment_control = self.I@mxyz
        moment_total = moment_control + moment_aero

        # Equations of motion
        uvwDot = (1/self.m) * (-np.cross(pqr, uvw) + force_total)
        pqrDot = self.I_inv @ (-np.cross(pqr, self.I@pqr) + moment_total)
        eulDot =  R_rates2Eul @ pqr
        xyzDot = R_b2i @ uvw
        dState = np.concatenate([uvwDot, pqrDot, eulDot, xyzDot])

        return dState
