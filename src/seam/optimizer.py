"""
SEAM Optimizer: Cost-based rewriting with rules R1-R6

Cost function: C(P) = steps + λ|seams| + Σ|stack|
Optimization: Greedy application of rewrites with ΔC < 0
"""
from typing import List, Tuple
from dataclasses import dataclass
import copy


def cost_function(ast: List, lambda_seam: float = 0.5) -> float:
    """
    Compute cost of SEAM program

    C(P) = steps + λ|seams| + Σ|stack|
    - steps: number of operations
    - |seams|: number of seam operators
    - |stack|: cumulative stack depth
    """
    from .parser import Seam, ConditionalSeam

    steps = len(ast)
    seam_count = sum(1 for expr in ast if isinstance(expr, Seam))

    # Estimate average stack depth (simplified)
    max_stack_depth = 0
    current_depth = 0
    for expr in ast:
        # Push operations increase depth
        if hasattr(expr, '__class__') and expr.__class__.__name__ in ['Load', 'ComputeEnergy']:
            current_depth += 1
        # Pop operations decrease depth
        elif hasattr(expr, '__class__') and expr.__class__.__name__ in ['DotProduct', 'Output']:
            current_depth = max(0, current_depth - 1)
        max_stack_depth = max(max_stack_depth, current_depth)

    # Conditional seams add branching cost
    for expr in ast:
        if isinstance(expr, ConditionalSeam):
            # Add cost of both branches (weighted by probability)
            true_cost = cost_function(expr.true_branch, lambda_seam)
            false_cost = cost_function(expr.false_branch, lambda_seam)
            steps += 0.5 * true_cost + 0.5 * false_cost

    return steps + lambda_seam * seam_count + max_stack_depth


@dataclass
class RewriteRule:
    """Represents a rewrite rule"""
    name: str
    pattern: callable  # Function that detects pattern
    replacement: callable  # Function that generates replacement


def rule_r1_seam_idempotent(ast: List) -> Tuple[bool, List]:
    """
    R1: § § → id
    Two consecutive seams cancel out
    """
    from .parser import Seam

    for i in range(len(ast) - 1):
        if isinstance(ast[i], Seam) and isinstance(ast[i + 1], Seam):
            # Found pattern, remove both seams
            new_ast = ast[:i] + ast[i+2:]
            return (True, new_ast)

    return (False, ast)


def rule_r2_seam_commute(ast: List) -> Tuple[bool, List]:
    """
    R2: § op → Φ(op) §
    Seam commutes with operations by applying duality map
    """
    from .parser import Seam, DotProduct

    # Simplified: only handle dot product for now
    for i in range(len(ast) - 1):
        if isinstance(ast[i], Seam) and isinstance(ast[i + 1], DotProduct):
            # Apply duality: § (·k) → (anti_dot k) §
            # For now, keep same (would need dual operation class)
            # This is a placeholder for the duality transformation
            return (False, ast)

    return (False, ast)


def rule_r5_seam_hoisting(ast: List) -> Tuple[bool, List]:
    """
    R5: Hoist seams outside loops/conditionals if beneficial
    """
    from .parser import ConditionalSeam, Seam

    for i, expr in enumerate(ast):
        if isinstance(expr, ConditionalSeam):
            # Check if both branches start with same seam
            if (len(expr.true_branch) > 0 and len(expr.false_branch) > 0 and
                isinstance(expr.true_branch[0], Seam) and
                isinstance(expr.false_branch[0], Seam)):
                # Hoist seam out
                new_true = expr.true_branch[1:]
                new_false = expr.false_branch[1:]
                new_conditional = ConditionalSeam(expr.threshold, new_true, new_false)
                new_ast = ast[:i] + [Seam(), new_conditional] + ast[i+1:]
                return (True, new_ast)

    return (False, ast)


def rule_r6_idempotent_elimination(ast: List) -> Tuple[bool, List]:
    """
    R6: § op § → op if op is self-dual
    """
    from .parser import Seam, Canonical

    for i in range(len(ast) - 2):
        if (isinstance(ast[i], Seam) and
            isinstance(ast[i + 2], Seam) and
            isinstance(ast[i + 1], Canonical)):
            # σ is self-dual, eliminate seams
            new_ast = ast[:i] + [ast[i + 1]] + ast[i+3:]
            return (True, new_ast)

    return (False, ast)


# All rewrite rules
REWRITE_RULES = [
    rule_r1_seam_idempotent,
    rule_r2_seam_commute,
    rule_r5_seam_hoisting,
    rule_r6_idempotent_elimination,
]


def optimize(ast: List, max_iterations: int = 100, lambda_seam: float = 0.5) -> List:
    """
    Optimize SEAM program using greedy rewriting

    Algorithm:
    1. Compute current cost C(P)
    2. Try each rewrite rule
    3. If any rule reduces cost (ΔC < 0), apply it
    4. Repeat until no improvements or max iterations
    """
    current_ast = copy.deepcopy(ast)
    current_cost = cost_function(current_ast, lambda_seam)

    for iteration in range(max_iterations):
        improved = False

        for rule in REWRITE_RULES:
            # Try applying rule
            changed, new_ast = rule(current_ast)

            if changed:
                new_cost = cost_function(new_ast, lambda_seam)
                delta_cost = new_cost - current_cost

                if delta_cost < 0:
                    # Accept improvement
                    current_ast = new_ast
                    current_cost = new_cost
                    improved = True
                    break  # Apply one rule per iteration

        if not improved:
            # No more improvements found
            break

    return current_ast


def estimate_speedup(original_ast: List, optimized_ast: List) -> float:
    """
    Estimate speedup from optimization

    Speedup = C(original) / C(optimized)
    """
    original_cost = cost_function(original_ast)
    optimized_cost = cost_function(optimized_ast)

    if optimized_cost == 0:
        return float('inf')

    return original_cost / optimized_cost
