"""The single-step controller (OBQAC) and the baselines it is compared against.

OBQAC, Eq. (7) of the paper:

    min_w  w^T W w
    s.t.   w = F(q_k)^T Delta_k / Ts - u_k        (distance to the unconstrained control)
           u_k = u_{k-1} + Delta u,  |Delta u| <= A Ts   (acceleration limits)
           u_k in U_c                              (velocity limits)

It is a quadratic programme in four variables with box constraints only. The paper solves it
with CasADi + ipopt; here the default is CasADi's QP interface, and `solver="closed_form"`
gives the exact answer for a diagonal W (the box sets are axis-aligned, so the projection
decouples: each channel is clipped independently, see NOTES.md).

The baselines are the ones the paper uses: an NMPC with horizon N_c, the same with integral
action, and an LQR designed on the model linearised about the reference (the paper's LTV
LQR). The DRL agent of the paper is not reproduced.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .model import NQ, NU, F, LeggedKinematics, wrap


# ----------------------------------------------------------------------------------- OBQAC
@dataclass
class OBQAC:
    """Eq. (7) with the error gain delta_e of Section III-D.

    The target displacement is not the whole gap to the next reference pose but

        Delta_k = (P_{k+1} - P_k) + delta_e (P_k - q_k) + K1 U1_k + K2 U2_k

    i.e. the reference's own motion plus a fraction delta_e of the tracking error (the paper's
    Eq. 10: e_{k+1} = (1 - delta_e) e_k), plus two optional leaky integral terms
    U1 = l1 U1 + Ts e, U2 = l2 U2 + Ts U1 inherited from the algebraic controllers of [1], [2].
    The defaults are the values of the paper's published MATLAB code: delta_e = 1 - 0.9802,
    K1 = K2 = 0.01, l1 = 0.47, l2 = 0.53. With delta_e = 1 (Eq. 7 taken literally) the
    heading loop under the acceleration limits is a saturated double integrator and oscillates;
    see NOTES.md.
    """
    model: LeggedKinematics
    delta_e: float = 1.0 - 0.9802
    K1: float = 0.01
    K2: float = 0.01
    l1: float = 0.47
    l2: float = 0.53
    W: np.ndarray = field(default_factory=lambda: np.eye(NU))
    solver: str = "qp"          # "qp" (CasADi qpsol), "ipopt" (as in the paper), "closed_form"
    name: str = "OBQAC"

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=float)
        self.reset()
        self._nlp = None
        if self.solver in ("qp", "ipopt"):
            self._build()

    def reset(self):
        self.u_prev = np.zeros(NU)
        self.U1 = np.zeros(NQ)
        self.U2 = np.zeros(NQ)
        self.solve_time = 0.0

    # --- the optimisation problem, built once ---------------------------------------
    def _build(self):
        import casadi as ca
        u = ca.SX.sym("u", NU)
        u_free = ca.SX.sym("u_free", NU)      # F^T Delta / Ts, a parameter
        w = u_free - u
        prob = {"x": u, "p": u_free, "f": w.T @ ca.DM(self.W) @ w}
        if self.solver == "qp":
            self._nlp = ca.qpsol("obqac", "qrqp", prob,
                                 {"print_time": False, "print_iter": False, "print_header": False,
                                  "print_info": False, "error_on_fail": True})
            self._plugin = "qrqp"
        else:
            self._nlp = ca.nlpsol("obqac", "ipopt", prob,
                                  {"print_time": False, "ipopt": {"print_level": 0, "sb": "yes"}})
            self._plugin = "ipopt"

    def bounds(self):
        """Intersection of U_c with the acceleration box around u_{k-1}."""
        lb = np.maximum(-self.model.u_max, self.u_prev - self.model.du_max)
        ub = np.minimum(self.model.u_max, self.u_prev + self.model.du_max)
        return lb, ub

    def target(self, q, p_now, p_next) -> np.ndarray:
        """Delta_k: where the robot is asked to be one sample from now, relative to q."""
        q = np.asarray(q, float)
        e = np.asarray(p_now, float) - q
        e[0] = wrap(e[0])
        self.U1 = self.l1 * self.U1 + self.model.ts * e
        self.U2 = self.l2 * self.U2 + self.model.ts * self.U1
        nu_r = np.asarray(p_next, float) - np.asarray(p_now, float)
        nu_r[0] = wrap(nu_r[0])
        return nu_r + self.delta_e * e + self.K1 * self.U1 + self.K2 * self.U2

    def __call__(self, q, p_now, p_next) -> np.ndarray:
        """Control for state q, the current reference pose P_k and the next one P_{k+1}."""
        t0 = time.perf_counter()
        q = np.asarray(q, float)
        u_free = F(q[0]).T @ self.target(q, p_now, p_next) / self.model.ts
        lb, ub = self.bounds()
        if self.solver == "closed_form":
            u = np.clip(u_free, lb, ub)
        else:
            sol = self._nlp(x0=np.clip(u_free, lb, ub), p=u_free, lbx=lb, ubx=ub)
            u = np.asarray(sol["x"]).ravel()
        self.solve_time = time.perf_counter() - t0
        self.u_prev = u
        return u


# ------------------------------------------------------------------------------------ NMPC
@dataclass
class NMPC:
    """Path-tracking NMPC on the model of Eq. (1), with the paper's stage and terminal costs.

        min  sum_i (P_{k+i} - q_{k+i})^T Q (.) + u_{k+i}^T R u_{k+i}  +  (P_{k+N} - q_{k+N})^T Qf (.)
        s.t. model, u in U_c, |Delta u| <= A Ts

    `integral=True` adds the paper's integral mode (iNMPC, [24]): the running sum of the
    tracking error is carried as an extra state and penalised with Qi.
    """
    model: LeggedKinematics
    horizon: int = 20
    # (theta, x, y, z) weights of the paper's published simulation code, Q = Qf, R = 0.1 I
    Q: np.ndarray = field(default_factory=lambda: np.diag([5.0, 10.0, 10.0, 0.001]))
    R: np.ndarray = field(default_factory=lambda: 0.1 * np.eye(NU))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([5.0, 10.0, 10.0, 0.001]))
    integral: bool = False
    Qi: np.ndarray = field(default_factory=lambda: 0.05 * np.eye(NQ))
    preview: bool = False       # False: hold P_{k+1} over the horizon, as the paper's code does
    name: str = "NMPC"

    def __post_init__(self):
        if self.integral and self.name == "NMPC":
            self.name = "iNMPC"
        self.u_prev = np.zeros(NU)
        self.err_int = np.zeros(NQ)
        self.solve_time = 0.0
        self._warm = None
        self._build()

    def reset(self):
        self.u_prev = np.zeros(NU)
        self.err_int = np.zeros(NQ)
        self._warm = None

    def _build(self):
        import casadi as ca
        N, ts = self.horizon, self.model.ts
        U = ca.SX.sym("U", NU, N)
        q0 = ca.SX.sym("q0", NQ)
        u_prev = ca.SX.sym("u_prev", NU)
        P = ca.SX.sym("P", NQ, N)
        e_int0 = ca.SX.sym("e_int0", NQ)
        Q, R, Qf, Qi = (ca.DM(m) for m in (self.Q, self.R, self.Qf, self.Qi))

        J, g = 0, []
        q, e_int, up = q0, e_int0, u_prev
        for i in range(N):
            u = U[:, i]
            g.append(u - up)                       # acceleration limits
            q = self.model.step_sym(q, u)
            e = P[:, i] - q
            e = ca.vertcat(ca.atan2(ca.sin(e[0]), ca.cos(e[0])), e[1:])
            Wm = Qf if i == N - 1 else Q
            J = J + e.T @ Wm @ e + u.T @ R @ u
            if self.integral:
                e_int = e_int + e * ts
                J = J + e_int.T @ Qi @ e_int
            up = u
        prob = {"x": ca.reshape(U, -1, 1), "p": ca.vertcat(q0, u_prev, e_int0, ca.reshape(P, -1, 1)),
                "f": J, "g": ca.vertcat(*g)}
        self._nlp = ca.nlpsol("nmpc", "ipopt", prob,
                              {"print_time": False,
                               "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 200,
                                         "tol": 1e-6, "warm_start_init_point": "yes"}})
        self._lbx = np.tile(-self.model.u_max, N)
        self._ubx = np.tile(self.model.u_max, N)
        self._lbg = np.tile(-self.model.du_max, N)
        self._ubg = np.tile(self.model.du_max, N)

    def __call__(self, q, P_horizon) -> np.ndarray:
        """P_horizon: (N, 4) reference poses for samples k+1 ... k+N."""
        t0 = time.perf_counter()
        P = np.asarray(P_horizon, dtype=float)
        if not self.preview:
            P = np.repeat(P[:1], self.horizon, axis=0)
        if P.shape[0] < self.horizon:                     # pad by holding the last pose
            P = np.vstack([P, np.repeat(P[-1:], self.horizon - P.shape[0], axis=0)])
        p = np.concatenate([np.asarray(q, float), self.u_prev, self.err_int, P[:self.horizon].T.ravel(order="F")])
        x0 = self._warm if self._warm is not None else np.tile(self.u_prev, self.horizon)
        sol = self._nlp(x0=x0, p=p, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg)
        U = np.asarray(sol["x"]).reshape(NU, self.horizon, order="F")
        self.solve_time = time.perf_counter() - t0
        u = U[:, 0].copy()
        self._warm = np.concatenate([U[:, 1:].ravel(order="F"), U[:, -1]])
        if self.integral:
            e = P[0] - self.model.step(q, u)
            e[0] = wrap(e[0])
            self.err_int += e * self.model.ts
        self.u_prev = u
        return u


# ------------------------------------------------------------------------------------- LQR
@dataclass
class LTVLQR:
    """LQR on the error dynamics e_{k+1} = e_k + Ts (nu_r - F(q) u), linearised about the
    reference each sample (hence time-varying). Infinite-horizon Riccati gain on
    (A = I, B = -Ts F(theta_ref)), feed-forward nu_r. Velocity limits are applied by
    saturation; the acceleration limits are NOT enforced, as in the paper's Fig. 5.
    """
    model: LeggedKinematics
    Q: np.ndarray = field(default_factory=lambda: np.diag([5.0, 10.0, 10.0, 0.001]))
    R: np.ndarray = field(default_factory=lambda: 0.1 * np.eye(NU))
    saturate: bool = True
    name: str = "LTV LQR"

    def __post_init__(self):
        self.u_prev = np.zeros(NU)
        self.solve_time = 0.0
        self._cache = {}

    def reset(self):
        self.u_prev = np.zeros(NU)

    def gain(self, theta_ref: float) -> np.ndarray:
        from scipy.linalg import solve_discrete_are
        key = round(float(theta_ref), 3)
        if key not in self._cache:
            A = np.eye(NQ)
            B = -self.model.ts * F(theta_ref)
            X = solve_discrete_are(A, B, self.Q, self.R)
            self._cache[key] = np.linalg.solve(B.T @ X @ B + self.R, B.T @ X @ A)
        return self._cache[key]

    def __call__(self, q, p_next, nu_r) -> np.ndarray:
        t0 = time.perf_counter()
        e = np.asarray(p_next, float) - np.asarray(q, float)
        e[0] = wrap(e[0])
        u_ff = F(q[0]).T @ np.asarray(nu_r, float)
        u = u_ff - self.gain(p_next[0]) @ e
        if self.saturate:
            u = np.clip(u, -self.model.u_max, self.model.u_max)
        self.solve_time = time.perf_counter() - t0
        self.u_prev = u
        return u
