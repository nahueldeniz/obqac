"""Plots for the examples and the notebook (matplotlib only)."""
from __future__ import annotations

import numpy as np


def plot_runs(runs, path, title: str = "", show_robot_every: int = 0):
    """Path, trajectories and the three metrics of Eq. (22) per sample, one figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title)
    pts = path.sample(800)
    a = ax[0, 0]
    a.plot(pts[0], pts[1], "k-", lw=2, label="path")
    for r in runs:
        a.plot(r.q[:, 1], r.q[:, 2], lw=1.2, label=r.name)
        a.plot(r.q[0, 1], r.q[0, 2], "o", ms=4, color=a.lines[-1].get_color())
    a.set_aspect("equal")
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    for k, (key, lab) in enumerate((("error", "deviation e_k"), ("effort_series", "control effort |u_k|"),
                                    ("rtf_series", "real-time factor"))):
        a = ax[0, 1] if k == 0 else ax[1, k - 1]
        for r in runs:
            a.plot(getattr(r, key), lw=1, label=r.name)
        a.set_xlabel("time step")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
        if key == "rtf_series":
            a.axhline(1.0, color="r", ls="--", lw=1)
            a.set_yscale("log")
    a = ax[0, 2]
    for r in runs:
        a.plot(r.u[:, 1], lw=1, label=f"{r.name} v_x")
        a.plot(r.u[:, 0], lw=1, ls="--", label=f"{r.name} omega")
    a.set_xlabel("time step")
    a.set_ylabel("control")
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    a = ax[1, 2]
    names = [r.name for r in runs]
    a.bar(names, [r.deviation for r in runs])
    a.set_ylabel("deviation  Xi")
    a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


def plot_summary_3d(summaries, title: str = ""):
    """The paper's Fig. 6: deviation, effort and real-time factor, one point per trial."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 5))
    a = fig.add_subplot(111, projection="3d")
    for name, rows in summaries.items():
        a.scatter([r["deviation"] for r in rows], [r["effort"] for r in rows], [r["rtf"] for r in rows],
                  label=name, s=40)
    a.set_xlabel("deviation Xi")
    a.set_ylabel("effort Pi")
    a.set_zlabel("real-time factor psi")
    a.legend(fontsize=8)
    a.set_title(title)
    return fig
