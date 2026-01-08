"""
SEAM Type System: Z₂-graded types for orientation tracking

Type structure:
  τ = (orientation, stack_type)
  orientation ∈ {Pos, Neg}  (Z₂ group)
  stack_type = [τ₁, τ₂, ..., τₙ]  (stack of types)
"""
from dataclasses import dataclass
from typing import List, Union
from enum import Enum


class Orientation(Enum):
    """Z₂ orientation: Pos or Neg"""
    POS = "Pos"
    NEG = "Neg"

    def flip(self) -> 'Orientation':
        """Flip orientation (Z₂ group operation)"""
        return Orientation.NEG if self == Orientation.POS else Orientation.POS

    def __mul__(self, other: 'Orientation') -> 'Orientation':
        """Compose orientations (group multiplication)"""
        if self == other:
            return Orientation.POS
        else:
            return Orientation.NEG


@dataclass
class SeamType:
    """Type for a value on the stack"""
    shape: str  # e.g., "Vec(N)", "Scalar", "Index"
    parity: Union[str, None] = None  # "Even", "Odd", or None (mixed)

    def __repr__(self):
        if self.parity:
            return f"{self.shape}[{self.parity}]"
        return self.shape


@dataclass
class ProgramState:
    """Type state during execution"""
    orientation: Orientation
    stack: List[SeamType]

    def __repr__(self):
        stack_str = " ".join(str(t) for t in self.stack)
        return f"{self.orientation.value} : {stack_str}"


class TypeError(Exception):
    """Raised when type checking fails"""
    pass


class TypeChecker:
    """Type checker for SEAM programs"""

    def __init__(self):
        self.state = ProgramState(Orientation.POS, [])

    def check_expr(self, expr) -> ProgramState:
        """Type check a single expression"""
        from .parser import (Load, Dup, Seam, ComputeEnergy, ProjectEven,
                            ProjectOdd, Canonical, DotProduct, ConditionalSeam,
                            TopK, Output, TypeAnnotation)

        if isinstance(expr, Load):
            # Load pushes a vector onto stack
            self.state.stack.append(SeamType("Vec(N)"))

        elif isinstance(expr, Dup):
            # Duplicate top of stack
            if not self.state.stack:
                raise TypeError("Stack underflow: cannot dup empty stack")
            self.state.stack.append(self.state.stack[-1])

        elif isinstance(expr, Seam):
            # Seam operator: flip orientation
            self.state.orientation = self.state.orientation.flip()

        elif isinstance(expr, ComputeEnergy):
            # α_±: consumes Vec(N), produces two Scalars (α₊, α₋)
            if not self.state.stack:
                raise TypeError("Stack underflow: α_± requires vector")
            vec_type = self.state.stack.pop()
            if not vec_type.shape.startswith("Vec"):
                raise TypeError(f"α_± requires Vec, got {vec_type.shape}")
            self.state.stack.append(SeamType("Scalar", "Plus"))
            self.state.stack.append(SeamType("Scalar", "Minus"))

        elif isinstance(expr, ProjectEven):
            # P+: consumes Vec(N), produces Vec(N)[Even]
            if not self.state.stack:
                raise TypeError("Stack underflow: P+ requires vector")
            vec_type = self.state.stack.pop()
            self.state.stack.append(SeamType(vec_type.shape, "Even"))

        elif isinstance(expr, ProjectOdd):
            # P-: consumes Vec(N), produces Vec(N)[Odd]
            if not self.state.stack:
                raise TypeError("Stack underflow: P- requires vector")
            vec_type = self.state.stack.pop()
            self.state.stack.append(SeamType(vec_type.shape, "Odd"))

        elif isinstance(expr, Canonical):
            # σ: consumes Vec(N), produces Vec(N)[Canonical]
            if not self.state.stack:
                raise TypeError("Stack underflow: σ requires vector")
            vec_type = self.state.stack.pop()
            self.state.stack.append(SeamType(vec_type.shape, "Canonical"))

        elif isinstance(expr, DotProduct):
            # ·: consumes Vec(N), produces Results
            if not self.state.stack:
                raise TypeError("Stack underflow: · requires vector")
            self.state.stack.pop()
            self.state.stack.append(SeamType("Results"))

        elif isinstance(expr, ConditionalSeam):
            # S_k*: branch based on threshold
            # Check both branches have same type signature
            saved_state = ProgramState(self.state.orientation, self.state.stack.copy())

            # Check true branch
            for e in expr.true_branch:
                self.check_expr(e)
            true_final = self.state.stack.copy()

            # Reset and check false branch
            self.state = ProgramState(saved_state.orientation, saved_state.stack.copy())
            for e in expr.false_branch:
                self.check_expr(e)
            false_final = self.state.stack.copy()

            # Branches should produce same stack shape
            if len(true_final) != len(false_final):
                raise TypeError(f"Conditional branches have different stack depths: "
                               f"{len(true_final)} vs {len(false_final)}")

        elif isinstance(expr, TopK):
            # top_k: consumes Results, produces Results
            if not self.state.stack:
                raise TypeError("Stack underflow: top_k requires results")
            result_type = self.state.stack.pop()
            if result_type.shape != "Results":
                raise TypeError(f"top_k requires Results, got {result_type.shape}")
            self.state.stack.append(SeamType("Results"))

        elif isinstance(expr, Output):
            # Output: consumes top of stack
            if not self.state.stack:
                raise TypeError("Stack underflow: output requires value")
            self.state.stack.pop()

        elif isinstance(expr, TypeAnnotation):
            # Type annotation: assert current state matches
            expected_orient = Orientation.POS if expr.orientation == "Pos" else Orientation.NEG
            if self.state.orientation != expected_orient:
                raise TypeError(f"Type mismatch: expected {expr.orientation}, "
                               f"got {self.state.orientation.value}")

        return self.state

    def check(self, ast: List) -> ProgramState:
        """Type check entire program"""
        for expr in ast:
            self.check_expr(expr)

        # Final check: stack should be empty or have output consumed
        if len(self.state.stack) > 1:
            raise TypeError(f"Program leaves {len(self.state.stack)} items on stack")

        return self.state
