"""
SEAM Interpreter: Executes SEAM programs

Stack-based execution with orientation tracking and seam semantics.
"""
import numpy as np
from typing import Any, List, Dict, Callable
from .types import Orientation


class RuntimeError(Exception):
    """Raised during SEAM program execution"""
    pass


class SeamInterpreter:
    """Interpreter for SEAM programs"""

    def __init__(self, variables: Dict[str, np.ndarray] = None):
        self.stack: List[Any] = []
        self.orientation = Orientation.POS
        self.variables = variables or {}
        self.seam_active = False  # Track if we're in seam context

        # Duality map Φ: maps operations to their duals
        self.duality_map = {
            'dot': 'anti_dot',  # Example: standard → antipodal
            'add': 'sub',
            'mul': 'div',
        }

    def push(self, value: Any):
        """Push value onto stack"""
        self.stack.append(value)

    def pop(self) -> Any:
        """Pop value from stack"""
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()

    def peek(self) -> Any:
        """Peek at top of stack without removing"""
        if not self.stack:
            raise RuntimeError("Stack empty")
        return self.stack[-1]

    def canonical(self, v: np.ndarray) -> np.ndarray:
        """
        Canonicalize vector to quotient representative σ(v)

        Algorithm: Normalize, find first nonzero element, flip if negative
        """
        # Normalize
        v_norm = v / (np.linalg.norm(v) + 1e-10)

        # Find first nonzero element
        for i, val in enumerate(v_norm):
            if abs(val) > 1e-10:
                # If negative, flip entire vector
                if val < 0:
                    v_norm = -v_norm
                break

        return v_norm

    def energy_partition(self, v: np.ndarray) -> tuple:
        """
        Compute α_± energy partition

        Returns: (α₊, α₋) where α₊ + α₋ = 1
        """
        v_can = self.canonical(v)

        # Compute even projection P+ = 0.5 * (v + σ(v))
        # If v is already canonical, P+ = v, P- = 0
        # If v is anticanonical, P+ = 0, P- = v

        # Check alignment with canonical
        alignment = np.dot(v / (np.linalg.norm(v) + 1e-10), v_can)

        if alignment > 0:  # Same hemisphere as canonical
            alpha_plus = alignment ** 2
            alpha_minus = 1 - alpha_plus
        else:  # Opposite hemisphere
            alpha_minus = alignment ** 2
            alpha_plus = 1 - alpha_minus

        return (alpha_plus, alpha_minus)

    def project_even(self, v: np.ndarray) -> np.ndarray:
        """Project to even parity subspace P+"""
        v_can = self.canonical(v)
        # P+ = 0.5 * (v + S_σ(v))
        # Where S_σ(v) = v_can if aligned, -v_can if anti-aligned
        v_norm = v / (np.linalg.norm(v) + 1e-10)
        alignment = np.dot(v_norm, v_can)

        if alignment > 0:
            return v_norm
        else:
            return np.zeros_like(v)

    def project_odd(self, v: np.ndarray) -> np.ndarray:
        """Project to odd parity subspace P-"""
        v_can = self.canonical(v)
        v_norm = v / (np.linalg.norm(v) + 1e-10)
        alignment = np.dot(v_norm, v_can)

        if alignment < 0:
            return -v_norm  # Flip to canonical hemisphere
        else:
            return np.zeros_like(v)

    def execute_expr(self, expr):
        """Execute single expression"""
        from .parser import (Load, Dup, Seam, ComputeEnergy, ProjectEven,
                            ProjectOdd, Canonical, DotProduct, ConditionalSeam,
                            TopK, Output, TypeAnnotation)

        if isinstance(expr, Load):
            # Load variable
            if expr.var not in self.variables:
                raise RuntimeError(f"Undefined variable: {expr.var}")
            self.push(self.variables[expr.var])

        elif isinstance(expr, Dup):
            # Duplicate top of stack
            self.push(self.peek())

        elif isinstance(expr, Seam):
            # Seam operator: flip orientation
            self.orientation = self.orientation.flip()
            self.seam_active = not self.seam_active

        elif isinstance(expr, ComputeEnergy):
            # Compute α_±
            v = self.pop()
            if not isinstance(v, np.ndarray):
                raise RuntimeError(f"α_± requires array, got {type(v)}")
            alpha_plus, alpha_minus = self.energy_partition(v)
            self.push(alpha_plus)
            self.push(alpha_minus)

        elif isinstance(expr, ProjectEven):
            # Project to even subspace
            v = self.pop()
            self.push(self.project_even(v))

        elif isinstance(expr, ProjectOdd):
            # Project to odd subspace
            v = self.pop()
            self.push(self.project_odd(v))

        elif isinstance(expr, Canonical):
            # Canonicalize
            v = self.pop()
            self.push(self.canonical(v))

        elif isinstance(expr, DotProduct):
            # Dot product with database
            v = self.pop()
            # This would search database; for now, return placeholder
            if expr.database in self.variables:
                db = self.variables[expr.database]
                # Simple dot product search (would use FAISS in real impl)
                scores = np.dot(db, v)
                self.push(scores)
            else:
                raise RuntimeError(f"Database not found: {expr.database}")

        elif isinstance(expr, ConditionalSeam):
            # Conditional seam S_k*
            alpha_minus = self.pop()
            alpha_plus = self.pop()
            alpha_max = max(alpha_plus, alpha_minus)

            # Choose branch based on threshold
            if alpha_max > expr.threshold:
                # Structure-dominated: use parity branch
                for e in expr.true_branch:
                    self.execute_expr(e)
            else:
                # Entropy-dominated: use quotient branch
                for e in expr.false_branch:
                    self.execute_expr(e)

        elif isinstance(expr, TopK):
            # Select top-k
            scores = self.pop()
            if isinstance(scores, np.ndarray):
                # Get indices of top k scores
                top_indices = np.argsort(scores)[-expr.k:][::-1]
                self.push(top_indices)
            else:
                self.push(scores)  # Pass through if not array

        elif isinstance(expr, Output):
            # Output (leave on stack for return)
            pass

        elif isinstance(expr, TypeAnnotation):
            # Type annotation (no runtime effect)
            pass

    def execute(self, ast: List) -> Any:
        """Execute SEAM program"""
        for expr in ast:
            self.execute_expr(expr)

        # Return top of stack (or None if empty)
        return self.stack[-1] if self.stack else None

    def __repr__(self):
        return f"SeamInterpreter(orientation={self.orientation.value}, stack_depth={len(self.stack)})"
