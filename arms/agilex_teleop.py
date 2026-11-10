"""Optional AgileX-style pose IK, independent of transport and hardware.

Adapts QuestArmTeleop's SE(3) pose objective with command continuity.
Uses SciPy bounded least squares instead of CasADi/IPOPT; not identical solver
numerics. Used by active teleop and dashboard Cartesian jog when enabled;
their output guards remain in force.
"""

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares


class AgilexTeleopIK:
    def __init__(self, solver, *, tolerance=1e-4):
        self.model = solver.urdf_model
        self.data = self.model.createData()
        self.frame = self.model.getFrameId(solver.end_effector_frame)
        self.lower = np.asarray(self.model.lowerPositionLimit)
        self.upper = np.asarray(self.model.upperPositionLimit)
        self.tolerance = float(tolerance)

    def solve(self, target, reference_deg, *, bounds_deg=None):
        lower, upper = (
            (self.lower, self.upper) if bounds_deg is None else np.radians(bounds_deg).T
        )
        previous = np.clip(np.radians(reference_deg), lower, upper)
        goal = pin.SE3(target[:3, :3], target[:3, 3])

        # Bounded least-squares nudges seeds away from a boundary. Preserve an
        # exactly stationary hand target, even when its hold is in a soft margin.
        pin.framesForwardKinematics(self.model, self.data, previous)
        if (
            np.linalg.norm(pin.log6(self.data.oMf[self.frame].inverse() * goal).vector)
            < 1e-10
        ):
            return np.degrees(previous)

        def residual(q):
            pin.framesForwardKinematics(self.model, self.data, q)
            error = pin.log6(self.data.oMf[self.frame].inverse() * goal).vector
            # Preserve the accepted command posture, not a measured tracking
            # offset or zero pose. At its FK target the residual is zero.
            return np.concatenate((2.0 * error, np.sqrt(.1) * (q - previous)))

        result = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            max_nfev=50,
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return None
        return np.degrees(result.x)
