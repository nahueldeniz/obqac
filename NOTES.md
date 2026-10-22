# Notes on this re-implementation

This is a Python + CasADi re-implementation of the controller and the simulated comparison of
Deniz and Auat Cheein, *IEEE Trans. Industrial Informatics* 22(2):729-739, 2026. It was written
from the paper and from the MATLAB code the paper links (`go1Sim.m`, `LAOBC.m`,
`nlmpcCasadi.m`, `nlmpcIntCasadi.m`), which is the reference for every choice below that the
paper's text leaves open. The numbers it produces are **not** the paper's numbers: different
language, solver, computer and random draws. The qualitative picture is the paper's, and that
is what the tests check.

## What the controller actually computes

Eq. (7) of the paper minimises the distance between the constrained control `u_k` and the
unconstrained one `F(q_k)^T Delta_k / Ts`, subject to the velocity box `U_c` and the
acceleration box `|u_k - u_{k-1}| <= A Ts`. The paper's Section III-D then introduces a gain
`delta_e in (0, 1]` on the tracking error so that the control "belongs to `U_c`", and its
Eq. (10) gives the closed loop `e_{k+1} = (1 - delta_e) e_k`.

The published MATLAB code (`LAOBC.solve`) is the concrete form of that gain:

    Delta_k = P_{k+1} - q_k - k (P_k - q_k) + K1 U1_k + K2 U2_k
            = (P_{k+1} - P_k) + (1 - k) (P_k - q_k) + K1 U1_k + K2 U2_k

with `k = 0.9802`, so `delta_e = 1 - k = 0.0198`, plus two leaky integral terms
`U1 = l1 U1 + Ts e`, `U2 = l2 U2 + Ts U1` with `K1 = K2 = 0.01`, `l1 = 0.47`, `l2 = 0.53`, inherited
from the algebraic controllers of refs. [1] and [2]. Those are the defaults of `OBQAC` here.
At `Ts = 0.05` s the proportional part is `delta_e / Ts = 0.4 /s` on top of the reference
velocity feed-forward, and the error decays as `0.98^k`: from 3 m to 5 cm in about 200
samples, which is what Fig. 2 of the paper shows. The integral terms are numerically tiny at
these gains (a steady error `e` adds `K1 Ts e / (1 - l1) ~ 0.001 e` to the step) and can be
switched off with `K1 = K2 = 0` without visible change.

**Eq. (7) taken literally (`delta_e = 1`) does not track under the paper's acceleration
limits.** With the whole gap demanded in one step, the commanded body-frame velocity saturates
in every channel and its direction swings with the heading; at 0.25 m/s^2 the robot cannot
re-orient its velocity as fast as its heading turns, and the position error never closes
(`results/literal_eq7.json`, `test_literal_eq7_does_not_converge`). The error gain is not a
tuning nicety, it is what makes the one-step controller a stable closed loop. This is stated
in the paper's stability section but easy to miss when reading Eq. (7) on its own.

## The QP has a closed form for a diagonal W

The feasible set of Eq. (7) is the intersection of two axis-aligned boxes, which is itself a
box, and for a diagonal `W` the objective separates by channel. The solution is therefore a
component-wise clip:

    u_k[i] = clip( u_free[i],  max(-U_c[i], u_{k-1}[i] - A[i] Ts),  min(U_c[i], u_{k-1}[i] + A[i] Ts) )

`OBQAC(solver="closed_form")` implements that; `solver="qp"` (CasADi's `qrqp`) and
`solver="ipopt"` (what the paper's code uses) solve the same problem numerically, and
`test_closed_form_equals_qp` checks they agree. The QP form matters when `W` couples channels,
or when you want to add constraints that are not boxes (the paper's Section III-F discusses
the mixed-integer view of the acceleration choice). The closed form is about a hundred times
faster than ipopt on this machine and gives an RTF of a few 10^-4.

## Baselines

- **NMPC / iNMPC**: same model, horizon `N_c`, stage cost `e^T Q e + u^T R u`, terminal `Qf`,
  the velocity and acceleration boxes as constraints, ipopt with warm start. Weights are the
  ones of the published simulation code, in (theta, x, y, z) order: `Q = Qf = diag(5, 10, 10,
  0.001)`, `R = 0.1 I` for the NMPC; `Q = Qf = diag(1, 2, 2, 0.001)`, `R = I` for the integral
  version. The paper's text quotes other matrices for the *field* experiments. As in the
  published code, the reference pose `P_{k+1}` is **held constant over the horizon**
  (`preview=False`); `preview=True` feeds the next `N_c` reference poses instead and tracks
  noticeably better, but that is not the comparison the paper made. Integral mode: the running
  sum of the tracking error is carried through the horizon and penalised with `Qi`.
- **LTV LQR**: infinite-horizon Riccati gain on the error dynamics `e_{k+1} = e_k + Ts (nu_r -
  F u)` linearised about the reference heading each sample, with velocity feed-forward and
  saturation to `U_c`. The acceleration limits are not enforced, which is the point the paper
  makes about it (Fig. 5). The paper does not say how its LTV LQR was designed.
- **DRL agent**: not reproduced. The paper's agent was trained in MATLAB's RL toolbox; the
  trained agent is in the paper's repository.

## Simulation protocol

Lemniscate of Eq. (21) (x amplitude 1.75 sqrt(2) m, y amplitude 2 m), reference speed
`nu_r = 0.75 m/s`, `Ts = 0.05 s`, one lap = 535 samples (the paper's figures run to ~530).
Initial pose 3 m from the path start in a random direction with a random heading, as the paper
describes; the published code's committed configuration instead starts 15 cm from the path.
Limits: `U_c = [-5pi/4, 5pi/4] x [-1.25, 1.25]^3`, accelerations `pi/2 rad/s^2` and
`0.25 m/s^2` (`--field` switches to the field limits `[-pi/2, pi/2] x [-0.5, 0.5]^3`).
Metrics are Eq. (22): effort `Pi`, deviation `Xi` (from Eq. 12) and real-time factor `psi`,
the latter measured as the controller's wall-clock time per sample over `Ts`.

Five trials, this machine (Python 3.12, CasADi 3.8):

| controller  | deviation Xi | effort Pi | RTF (mean) | RTF (max) | error at end of lap |
|-------------|-------------:|----------:|-----------:|----------:|--------------------:|
| OBQAC       | 0.068 | 0.041 | 0.0017 | 0.005 | 0.001 |
| NMPC N=20   | 0.067 | 0.043 | 0.171  | 0.39  | 0.42  |
| iNMPC N=20  | 0.078 | 0.037 | 0.156  | 0.27  | 0.93  |
| LTV LQR     | 0.032 | 0.052 | 0.003  | 0.53* | 0.04  |
| NMPC N=1    | 0.094 | 0.048 | 0.028  | 0.06  | 0.90  |
| NMPC N=2    | 0.097 | 0.049 | 0.036  | 0.05  | 1.25  |
| NMPC N=10   | 0.073 | 0.045 | 0.067  | 0.13  | 0.46  |

\* first call, scipy import.

What reproduces: OBQAC and NMPC(20) track comparably, at two orders of magnitude difference in
compute (the paper: 0.17 vs 1.5 RTF, one order; its NMPC ran in MATLAB); the short-horizon
NMPCs of Fig. 4 do not converge within the lap; the integral NMPC is slower to reach the path;
the LQR has the smallest deviation but changes its commands up to 1 rad per sample where the
limit is 0.079 rad. What does not: absolute RTF values (no NMPC here exceeds 1), and the NMPC's
residual error at the end of the lap (0.4 here against ~0.1 in Fig. 2), which is the
held-reference formulation lagging on the curved path.

## Field data

The paper's repository also contains the field logs (`go1-infty*.mat`, `go1-circular*.mat`,
`go1-rectangular*.mat`). They are not needed here and are not copied.
