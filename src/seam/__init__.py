"""
SEAM: Esoteric Language for Non-Orientable Computing

A stack-based language with orientation tracking and seam operators
for quotient manifolds like ℝPⁿ⁻¹.
"""

from .parser import parse
from .interpreter import SeamInterpreter
from .types import TypeChecker, Orientation
from .optimizer import optimize, cost_function

__version__ = "0.1.0"
__all__ = ["parse", "SeamInterpreter", "TypeChecker", "Orientation", "optimize", "cost_function"]
