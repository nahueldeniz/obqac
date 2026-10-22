"""Closed-loop simulation and the three metrics of Eq. (22).

    Pi   control effort     (1/K) sqrt( sum_i |u_i|^2 )
    Xi   deviation          (1/K) sqrt( sum_i d_theta,i^2 + d_s,i^2 )     (Eq. 12)
    psi  real-time factor   (1/(Ts K)) sum_i xi_i,   xi_i = time to compute u_i

The initial pose is drawn as in the paper: 3 m from the path start (random direction) with a
random heading in [-pi, pi]. Trials use a fixed sequence of seeds so results are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import LTVLQR, NMPC, OBQAC
from .model import LeggedKinematics, wrap
from .path import Path, Reference


@dataclass
class Run:
    name: str
    t: np.ndarray
    q: np.ndarray            # (K+1, 4)
    p: np.ndarray            # (K, 4) reference poses
    u: np.ndarray            # (K, 4)
    solve_time: np.ndarray   # (K,)
    ts: float
    extra: dict = field(default_factory=dict)

    # --- Eq. (22) ---------------------------------------------------------------------
    @property
    def error(self) -> np.ndarray:
        """e_i of Eq. (11): heading error and Euclidean distance to the reference, combined."""
        d_theta = wrap(self.p[:, 0] - self.q[1:, 0])
        d_s = np.linalg.norm(self.p[:, 1:] - self.q[1:, 1:], axis=1)
        return np.sqrt(d_theta ** 2 + d_s ** 2)

    @property
    def effort(self) -> float:
        return float(np.sqrt(np.sum(self.u ** 2)) / len(self.u))

    @property
    def deviation(self) -> float:
        return float(np.sqrt(np.sum(self.error ** 2)) / len(self.u))

    @property
    def rtf(self) -> float:
        return float(np.mean(self.solve_time) / self.ts)

    @property
    def rtf_series(self) -> np.ndarray:
        return self.solve_time / self.ts

    @property
    def effort_series(self) -> np.ndarray:
        return np.linalg.norm(self.u, axis=1)

    def summary(self) -> dict:
        return {"controller": self.name, "effort": self.effort, "deviation": self.deviation,
                "rtf": self.rtf, "rtf_max": float(self.rtf_series.max()),
                "final_error": float(self.error[-1]), "steps": int(len(self.u))}


def initial_pose(path: Path, ref: Reference, rng: np.random.Generator,
                 d0: float = 3.0) -> np.ndarray:
    p0 = ref.pose
    ang = rng.uniform(-np.pi, np.pi)
    theta0 = rng.uniform(-np.pi, np.pi)
    return np.array([theta0, p0[1] + d0 * np.cos(ang), p0[2] + d0 * np.sin(ang), path.z])


def simulate(controller, path: Path, model: LeggedKinematics, speed: float = 0.75,
             steps: int | None = None, seed: int = 0, d0: float = 3.0,
             q0=None) -> Run:
    """One trial. `controller` is an OBQAC, NMPC or LTVLQR instance."""
    rng = np.random.default_rng(seed)
    ref = Reference(path, speed, model.ts)
    K = steps or ref.steps_per_lap()
    q = initial_pose(path, ref, rng, d0) if q0 is None else np.asarray(q0, float)
    controller.reset()

    Q, P, U, T = [q.copy()], [], [], []
    for _ in range(K):
        p_now, p_next = ref.pose, ref.pose_at(ref.s + speed * model.ts)
        if isinstance(controller, OBQAC):
            u = controller(q, p_now, p_next)
        elif isinstance(controller, NMPC):
            u = controller(q, ref.horizon(controller.horizon))
        elif isinstance(controller, LTVLQR):
            u = controller(q, p_next, ref.velocity())
        else:
            raise TypeError(type(controller))
        T.append(controller.solve_time)
        q = model.step(q, u)
        ref.advance()
        Q.append(q.copy())
        P.append(p_next)
        U.append(np.asarray(u, float))
    t = np.arange(K + 1) * model.ts
    return Run(controller.name, t, np.array(Q), np.array(P), np.array(U), np.array(T), model.ts)


def trials(make_controller, path: Path, model: LeggedKinematics, n: int = 5, speed: float = 0.75,
           steps: int | None = None, d0: float = 3.0, seed0: int = 0) -> list[Run]:
    """n trials with seeds seed0, seed0+1, ... ; the same seeds give the same initial poses to
    every controller, so the comparison is paired."""
    ctrl = make_controller()
    return [simulate(ctrl, path, model, speed, steps, seed0 + i, d0) for i in range(n)]


def aggregate(runs: list[Run]) -> dict:
    s = [r.summary() for r in runs]
    keys = ("effort", "deviation", "rtf", "rtf_max", "final_error")
    out = {"controller": runs[0].name, "trials": len(runs)}
    for k in keys:
        vals = np.array([x[k] for x in s])
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out
