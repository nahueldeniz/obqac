"""Reference trajectories and the moving reference pose P_k of Eq. (2).

A path is a closed planar curve parameterised by an angle. The reference generator walks
along it at a constant ground speed nu_r (the paper's Theta_k is "designed such that" the
step between consecutive reference points is nu_r Ts), and returns the pose
P_k = (p_theta, p_x, p_y, p_z) with p_theta the direction of travel, exactly as Eq. (2)
defines it from the next point on the path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import wrap


@dataclass
class Path:
    xy: Callable[[np.ndarray], np.ndarray]   # angle -> (2, n) array
    z: float = 0.29                          # robot height, constant (the paper's p_z)
    n_table: int = 20000

    def __post_init__(self):
        th = np.linspace(0.0, 2 * np.pi, self.n_table + 1)
        pts = self.xy(th)
        seg = np.hypot(np.diff(pts[0]), np.diff(pts[1]))
        self._th = th
        self._s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self._s[-1])

    def point_at_length(self, s: float) -> np.ndarray:
        s = s % self.length
        th = np.interp(s, self._s, self._th)
        return self.xy(np.atleast_1d(th))[:, 0]

    def sample(self, n: int = 600) -> np.ndarray:
        return self.xy(np.linspace(0.0, 2 * np.pi, n))

    def distance(self, xy) -> float:
        pts = self.sample(4000)
        return float(np.min(np.hypot(pts[0] - xy[0], pts[1] - xy[1])))


def lemniscate(ax: float = 1.75 * np.sqrt(2), ay: float = np.sqrt(32), centre=(0.0, 0.0),
               z: float = 0.29) -> Path:
    """Eq. (21) of the paper. The default amplitudes are the paper's; x spans 4.95 m, y 4 m."""
    cx, cy = centre

    def xy(th):
        d = np.sin(th) ** 2 + 1
        return np.vstack([cx + ax * np.cos(th) / d, cy + ay * np.cos(th) * np.sin(th) / d])
    return Path(xy, z)


def circle(radius: float = 2.0, centre=(0.0, 0.0), z: float = 0.29) -> Path:
    cx, cy = centre

    def xy(th):
        return np.vstack([cx + radius * np.cos(th), cy + radius * np.sin(th)])
    return Path(xy, z)


def rectangle(width: float = 5.0, height: float = 4.0, centre=(0.0, 0.0), z: float = 0.29,
              corner: float = 0.15) -> Path:
    """Rectangle with slightly rounded corners so the tangent (p_theta) is defined everywhere."""
    cx, cy = centre
    w, h, r = width / 2 - corner, height / 2 - corner, corner
    straight = 2 * (2 * w + 2 * h)
    arcs = 2 * np.pi * r
    total = straight + arcs

    def one(s):
        s = s % total
        # walk anticlockwise from (w+r, -h): right side up, top leftwards, left down, bottom right
        legs = [(2 * h, lambda t: (w + r, -h + t)),
                (np.pi * r / 2, lambda t: (w + r * np.cos(t / r), h + r * np.sin(t / r))),
                (2 * w, lambda t: (w - t, h + r)),
                (np.pi * r / 2, lambda t: (-w + r * np.cos(np.pi / 2 + t / r), h + r * np.sin(np.pi / 2 + t / r))),
                (2 * h, lambda t: (-w - r, h - t)),
                (np.pi * r / 2, lambda t: (-w + r * np.cos(np.pi + t / r), -h + r * np.sin(np.pi + t / r))),
                (2 * w, lambda t: (-w + t, -h - r)),
                (np.pi * r / 2, lambda t: (w + r * np.cos(3 * np.pi / 2 + t / r), -h + r * np.sin(3 * np.pi / 2 + t / r)))]
        for length, f in legs:
            if s <= length:
                x, y = f(s)
                return cx + x, cy + y
            s -= length
        x, y = legs[-1][1](legs[-1][0])
        return cx + x, cy + y

    def xy(th):
        th = np.atleast_1d(th)
        pts = np.array([one(t / (2 * np.pi) * total) for t in th])
        return pts.T
    return Path(xy, z)


PATHS = {"lemniscate": lemniscate, "circle": circle, "rectangle": rectangle}


class Reference:
    """P_k of Eq. (2): the point nu_r Ts further along the path each sample, heading along it."""

    def __init__(self, path: Path, speed: float = 0.75, ts: float = 0.05, s0: float = 0.0):
        self.path, self.speed, self.ts = path, float(speed), float(ts)
        self.s = float(s0)

    def pose_at(self, s: float) -> np.ndarray:
        p = self.path.point_at_length(s)
        p1 = self.path.point_at_length(s + self.speed * self.ts)
        theta = np.arctan2(p1[1] - p[1], p1[0] - p[0])
        return np.array([theta, p[0], p[1], self.path.z])

    @property
    def pose(self) -> np.ndarray:
        return self.pose_at(self.s)

    def horizon(self, n: int) -> np.ndarray:
        """Reference poses for the next n samples (n, 4), starting one step ahead."""
        return np.array([self.pose_at(self.s + i * self.speed * self.ts) for i in range(1, n + 1)])

    def velocity(self) -> np.ndarray:
        """nu_r of Eq. (3): rate of change of the reference over the next sample, global frame."""
        p0, p1 = self.pose, self.pose_at(self.s + self.speed * self.ts)
        d = p1 - p0
        d[0] = wrap(d[0])
        return d / self.ts

    def advance(self) -> np.ndarray:
        self.s += self.speed * self.ts
        return self.pose

    def steps_per_lap(self) -> int:
        return int(np.floor(self.path.length / (self.speed * self.ts)))
