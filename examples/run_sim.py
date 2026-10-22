#!/usr/bin/env python3
"""Path tracking with the single-step controller and the paper's baselines.

  python examples/run_sim.py                                   # OBQAC vs NMPC(20) on the lemniscate, 5 trials
  python examples/run_sim.py --controllers obqac nmpc inmpc lqr --nc 20
  python examples/run_sim.py --controllers nmpc --nc 1 2 10    # the short horizons of Fig. 4
  python examples/run_sim.py --path circle --field              # the field experiments' limits
  python examples/run_sim.py --delta-e 1.0                      # Eq. (7) taken literally (see NOTES.md)
  python examples/run_sim.py --trials 1 --plot                  # one trial, with the figure

Results (metrics per trial and their mean) go to results/<name>.json; --plot also writes a .png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obqac import LTVLQR, NMPC, OBQAC, PATHS, LeggedKinematics, aggregate, simulate  # noqa: E402

FIELD = dict(u_max=(np.pi / 2, 0.5, 0.5, 0.5), a_max=(np.pi / 2, 0.25, 0.25, 0.25))
SIM = dict(u_max=(5 * np.pi / 4, 1.25, 1.25, 1.25), a_max=(np.pi / 2, 0.25, 0.25, 0.25))


def make(name: str, model: LeggedKinematics, args, nc: int):
    if name == "obqac":
        return OBQAC(model, delta_e=args.delta_e, solver=args.solver)
    if name == "nmpc":
        return NMPC(model, horizon=nc, name=f"NMPC N={nc}")
    if name == "inmpc":
        return NMPC(model, horizon=nc, integral=True, Q=np.diag([1, 2, 2, 0.001]),
                    Qf=np.diag([1, 2, 2, 0.001]), R=np.eye(4), name=f"iNMPC N={nc}")
    if name == "lqr":
        return LTVLQR(model)
    raise SystemExit(f"unknown controller {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controllers", nargs="+", default=["obqac", "nmpc"],
                   choices=["obqac", "nmpc", "inmpc", "lqr"])
    p.add_argument("--nc", type=int, nargs="+", default=[20], help="NMPC horizon(s)")
    p.add_argument("--path", default="lemniscate", choices=list(PATHS))
    p.add_argument("--speed", type=float, default=0.75, help="reference speed nu_r [m/s]")
    p.add_argument("--ts", type=float, default=0.05)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d0", type=float, default=3.0, help="initial distance to the path [m]")
    p.add_argument("--steps", type=int, default=None, help="samples per trial (default: one lap)")
    p.add_argument("--field", action="store_true", help="velocity limits of the field experiments")
    p.add_argument("--delta-e", type=float, default=1 - 0.9802, help="OBQAC error gain (Section III-D)")
    p.add_argument("--solver", default="qp", choices=["qp", "ipopt", "closed_form"])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", default=None, help="results name (default from the options)")
    args = p.parse_args()

    model = LeggedKinematics(args.ts, **(FIELD if args.field else SIM))
    path = PATHS[args.path]()
    name = args.out or f"{args.path}_{'field' if args.field else 'sim'}"
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)

    runs_by, summary = {}, {}
    for c in args.controllers:
        for nc in (args.nc if c in ("nmpc", "inmpc") else [None]):
            ctrl = make(c, model, args, nc)
            runs = [simulate(ctrl, path, model, args.speed, args.steps, args.seed + i, args.d0)
                    for i in range(args.trials)]
            runs_by[ctrl.name] = runs
            summary[ctrl.name] = {"trials": [r.summary() for r in runs], "mean": aggregate(runs)}
            m = summary[ctrl.name]["mean"]
            print(f"{ctrl.name:14s} deviation {m['deviation']:.4f}  effort {m['effort']:.4f}  "
                  f"RTF {m['rtf']:.4f} (max {m['rtf_max']:.3f})  final error {m['final_error']:.3f}")
    (out / f"{name}.json").write_text(json.dumps({"args": vars(args), "results": summary}, indent=1))
    print(f"-> {out / f'{name}.json'}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        from obqac.viz import plot_runs
        fig = plot_runs([runs[0] for runs in runs_by.values()], path,
                        f"{args.path}, trial 0 of {args.trials}")
        fig.savefig(out / f"{name}.png", dpi=130)
        print(f"-> {out / f'{name}.png'}")


if __name__ == "__main__":
    main()
