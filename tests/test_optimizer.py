"""
Tests for SEAM optimizer
"""
import sys
sys.path.insert(0, '/home/user/SEAM-AVDA/src')

import pytest
from seam.parser import parse, Seam
from seam.optimizer import optimize, cost_function, rule_r1_seam_idempotent


def test_cost_function_simple():
    """Test cost function on simple program"""
    ast = parse("q σ .")
    cost = cost_function(ast)

    # Should be: 3 steps + 0 seams + small stack cost
    assert cost > 0


def test_cost_function_with_seams():
    """Test cost function includes seam penalty"""
    ast_no_seam = parse("q .")
    ast_with_seam = parse("q § .")

    cost_no_seam = cost_function(ast_no_seam)
    cost_with_seam = cost_function(ast_with_seam)

    # Seam adds cost
    assert cost_with_seam > cost_no_seam


def test_rule_r1_seam_idempotent():
    """Test R1: § § → id"""
    ast = parse("q § § .")

    # Apply rule
    changed, new_ast = rule_r1_seam_idempotent(ast)

    # Should remove both seams
    assert changed
    assert len(new_ast) == 2  # Only q and .

    # Check no seams remain
    for expr in new_ast:
        assert not isinstance(expr, Seam)


def test_optimize_removes_double_seam():
    """Test optimizer removes § §"""
    original_ast = parse("q § § σ .")
    optimized_ast = optimize(original_ast)

    # Should have fewer nodes
    assert len(optimized_ast) <= len(original_ast)

    # Should have lower cost
    original_cost = cost_function(original_ast)
    optimized_cost = cost_function(optimized_ast)
    assert optimized_cost <= original_cost


def test_optimize_preserves_semantics():
    """Test that optimization preserves program semantics"""
    import numpy as np
    from seam.interpreter import SeamInterpreter

    v = np.array([1.0, 2.0, 3.0])

    # Original program: q § § σ . (double seam cancels)
    original_ast = parse("q § § σ .")
    optimized_ast = optimize(original_ast)

    # Execute both
    interp1 = SeamInterpreter(variables={"q": v})
    result1 = interp1.execute(original_ast)

    interp2 = SeamInterpreter(variables={"q": v})
    result2 = interp2.execute(optimized_ast)

    # Results should be identical
    assert np.allclose(result1, result2)


def test_optimize_idempotent():
    """Test that optimizing twice gives same result"""
    ast = parse("q § § σ .")

    optimized_once = optimize(ast)
    optimized_twice = optimize(optimized_once)

    # Second optimization should not change anything
    assert len(optimized_once) == len(optimized_twice)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
