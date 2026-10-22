"""Kinematic model of a legged robot driven through its built-in gait controller (Eq. 1).

State  q = (theta, x, y, z): heading and position in the global frame.
Input  u = (omega, v_x, v_y, v_z): yaw rate and body-frame velocities, the quantities a
commercial quadruped's low-level controller accepts.

    q_{k+1} = q_k + F(q_k) u_k Ts,      F(q) = blkdiag(1, R(theta), 1)

F is orthogonal, F^T F = I, which is the whole reason the controller in this package exists:
inverting the kinematics costs one transpose.
"""
from __future__ import annotations

import numpy as np

NQ = 4
NU = 4


def F(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, c, -s, 0.0],
                     [0.0, s, c, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def F_sym(theta):
    """CasADi version of F for use inside symbolic graphs."""
    import casadi as ca
    c, s = ca.cos(theta), ca.sin(theta)
    return ca.vertcat(ca.horzcat(1, 0, 0, 0),
                      ca.horzcat(0, c, -s, 0),
                      ca.horzcat(0, s, c, 0),
                      ca.horzcat(0, 0, 0, 1))


def wrap(angle):
    """Wrap to (-pi, pi]. Works on floats and arrays."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class LeggedKinematics:
    """Discrete-time model of Eq. (1), plus the box sets of Section III."""

    def __init__(self, ts: float = 0.05,
                 u_max=(5 * np.pi / 4, 1.25, 1.25, 1.25),
                 a_max=(np.pi / 2, 0.25, 0.25, 0.25)):
        self.ts = float(ts)
        self.u_max = np.asarray(u_max, dtype=float)   # U_c: symmetric box on u
        self.a_max = np.asarray(a_max, dtype=float)   # acceleration limits, |u_k - u_{k-1}| <= a_max Ts

    @property
    def du_max(self) -> np.ndarray:
        """Largest allowed change of u between two samples."""
        return self.a_max * self.ts

    def step(self, q, u) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        u = np.asarray(u, dtype=float)
        qn = q + F(q[0]) @ u * self.ts
        qn[0] = wrap(qn[0])
        return qn

    def step_sym(self, q, u):
        return q + F_sym(q[0]) @ u * self.ts

    def unconstrained_control(self, q, p_next) -> np.ndarray:
        """The u that lands exactly on p_next in one sample: F(q)^T (p_next - q) / Ts."""
        q = np.asarray(q, dtype=float)
        d = np.asarray(p_next, dtype=float) - q
        d[0] = wrap(d[0])
        return F(q[0]).T @ d / self.ts
