import jax as jax
import jax.numpy as np

class Quadcopter():
    def __init__(self):
        self.g = 9.807  # gravity (m / s**2)
        self.m = 2.5    # mass (kg)
        self.I = np.eye(3)  # Inertia tensor
        self.I_inv = np.linalg.inv(self.I)

    def dynamics(self, state, control, windNed=np.zeros(3)):
        """
        Dynamic function for quadcopter object
        xDot = f(x,u)
        x = [u,v,w,p,q,r,phi,theta,psi,x,y,z]
        u = [-fz,mx,my,mz]   (all mass/inertia normalized, accelerations)
        """
        # Unpack inputs
        uvw = state[0:3]
        pqr = state[3:6]
        phi,theta,psi = state[6:9]
        xyz = state[9:12]       # Inertial positions (ie. north, east, down)
        thrust = control[0]     # vertical acceleration (thrust)
        mxyz = control[1:4]     # angular accelerations (moments)

        # Body to Inertial Rotation matrix
        R1 = np.array([[1,0,0],[0,np.cos(phi),np.sin(phi)],[0,-np.sin(phi),np.cos(phi)]])
        R2 = np.array([[np.cos(theta), 0, -np.sin(theta)],[0, 1, 0],[np.sin(theta), 0, np.cos(theta)]])
        R3 = np.array([[np.cos(psi), np.sin(psi), 0],[-np.sin(psi), np.cos(psi), 0],[0, 0, 1]])
        R = R1@R2@R3

        # Angular rates to euler rates
        R_angRates = np.array([
            [1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])

        # Get total force / moments in body axis
        force_control = self.m*np.array([0,0,-thrust])
        force_drag = -0.1*(uvw+R.T@windNed)       # TODO: Pick decent values, and move to init
        force_gravity = self.m*self.g*R[:,2]
        force_total = force_control + force_drag + force_gravity

        moment_control = self.I@mxyz
        moment_drag = -0.1*pqr      # TODO: Pick decent values, and move to init
        moment_total = moment_control + moment_drag

        # Equations of motion
        uvwDot = (1/self.m) * (-np.cross(pqr, uvw) + force_total)
        pqrDot = self.I_inv @ (-np.cross(pqr, self.I@pqr) + moment_total)
        eulDot =  R_angRates @ pqr
        xyzDot = R@uvw
        xDot = np.concatenate([uvwDot, pqrDot, eulDot, xyzDot])

        return xDot
