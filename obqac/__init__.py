"""obqac: a single-step optimisation-based kinematic controller for legged robots.

Python + CasADi implementation of Deniz and Auat Cheein, "A Single-Step Optimization-Based
Kinematic Controller for Real-Time Path Tracking of Legged Robots", IEEE Transactions on
Industrial Informatics, 22(2):729-739, 2026. doi:10.1109/TII.2025.3609176
"""
from .controller import LTVLQR, NMPC, OBQAC
from .model import LeggedKinematics
from .path import PATHS, Reference, circle, lemniscate, rectangle
from .sim import Run, aggregate, simulate, trials

__all__ = ["OBQAC", "NMPC", "LTVLQR", "LeggedKinematics", "PATHS", "Reference", "circle",
           "lemniscate", "rectangle", "Run", "aggregate", "simulate", "trials"]
__version__ = "0.1.0"
