# obqac

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahueldeniz/obqac/blob/main/notebooks/obqac_demo.ipynb)

One optimisation step instead of twenty: a path-tracking controller for legged robots that
runs on whatever computer is already on the robot.

Python + CasADi implementation of

> N. N. Deniz and F. A. Auat Cheein, **"A Single-Step Optimization-Based Kinematic Controller
> for Real-Time Path Tracking of Legged Robots"**, *IEEE Transactions on Industrial
> Informatics*, 22(2):729-739, 2026.
> [doi:10.1109/TII.2025.3609176](https://doi.org/10.1109/TII.2025.3609176)

A commercial quadruped already knows how to walk: its built-in controller takes a forward
speed, a lateral speed and a yaw rate and turns them into gaits. Seen from above, that makes
the robot a kinematic model `q_{k+1} = q_k + F(q_k) u_k Ts` whose matrix `F` is a rotation,
so the control that lands on the next reference pose in one sample is one transpose away:
`u = F(q)^T Delta / Ts`. The catch is that this `u` ignores the robot's velocity and
acceleration limits. The controller keeps the one-step idea and adds the limits back as a
quadratic programme in four variables: find the admissible `u` closest to the one-step
answer, with the velocity box and the acceleration box as constraints. A gain on the tracking
error (Section III-D of the paper) makes the closed loop a contraction, `e_{k+1} = (1 -
delta_e) e_k`. No horizon, no training, two parameters.

## Install

```bash
git clone https://github.com/nahueldeniz/obqac
cd obqac
pip install -r requirements.txt
```

## Run

```bash
python examples/run_sim.py                                     # OBQAC vs NMPC(20), lemniscate, 5 trials
python examples/run_sim.py --controllers obqac nmpc inmpc lqr  # the paper's Figs. 2, 3 and 5
python examples/run_sim.py --controllers nmpc --nc 1 2 10      # Fig. 4: short horizons
python examples/run_sim.py --field --speed 0.35 --path circle  # the field experiments' limits
python examples/run_sim.py --delta-e 1.0                        # Eq. (7) without the error gain
python examples/run_sim.py --trials 1 --plot                    # one trial with the figure
python -m pytest tests -q
```

Metrics per trial and their mean go to `results/<name>.json`; `--plot` adds a `.png`.

```python
from obqac import OBQAC, LeggedKinematics, lemniscate, simulate

model = LeggedKinematics(ts=0.05)
run = simulate(OBQAC(model), lemniscate(), model, seed=0)
print(run.deviation, run.effort, run.rtf)
```

`OBQAC(model, delta_e=..., solver="qp" | "ipopt" | "closed_form")`. For a diagonal weight the
QP is a component-wise clip, and `closed_form` returns exactly that (see `NOTES.md`).

## What is here

```
obqac/model.py        Eq. (1): the kinematic model, F(q), the velocity and acceleration boxes
obqac/path.py         Eq. (2), (21): lemniscate, circle, rectangle; the moving reference pose
obqac/controller.py   OBQAC (Eq. 7 with the error gain of III-D); NMPC / iNMPC; LTV LQR
obqac/sim.py          closed loop, random initial poses, the metrics of Eq. (22)
examples/run_sim.py   the comparison of Section IV-A
notebooks/            Colab demo
tests/                14 tests: model, reference, QP = clip, contraction rate, baselines
results/              metrics and figures produced by run_sim.py
NOTES.md              what was reproduced, what was not, and the reading of Eq. (7) that works
```

The paper's own MATLAB code (controller, NMPC baselines, trained RL agent, field logs) is at
[github.com/nahueldeniz/A-Single-Step-Optimization-Based-Kinematic-Controller-for-Real-Time-Path-Tracking-of-Legged-Robots](https://github.com/nahueldeniz/A-Single-Step-Optimization-Based-Kinematic-Controller-for-Real-Time-Path-Tracking-of-Legged-Robots).
This repository is the free-tools version of the simulated comparison; it does not reproduce
the reinforcement-learning agent or the field experiments.

## Licence

MIT. If you use it, cite the paper (`CITATION.cff`).
