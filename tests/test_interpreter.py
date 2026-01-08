"""
Tests for SEAM interpreter
"""
import sys
sys.path.insert(0, '/home/user/SEAM-AVDA/src')

import pytest
import numpy as np
from seam.interpreter import SeamInterpreter
from seam.parser import parse


def test_canonical_positive():
    """Test canonical representative for positive-first vector"""
    interp = SeamInterpreter()
    v = np.array([1.0, 2.0, 3.0])
    v_can = interp.canonical(v)

    # Should be normalized
    assert np.isclose(np.linalg.norm(v_can), 1.0)

    # First element should be positive
    assert v_can[0] > 0


def test_canonical_negative():
    """Test canonical representative for negative-first vector"""
    interp = SeamInterpreter()
    v = np.array([-1.0, 2.0, 3.0])
    v_can = interp.canonical(v)

    # First element should be flipped to positive
    assert v_can[0] > 0


def test_canonical_antipodal_equivalence():
    """Test that v and -v have same canonical representative"""
    interp = SeamInterpreter()
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = -v1

    v1_can = interp.canonical(v1)
    v2_can = interp.canonical(v2)

    assert np.allclose(v1_can, v2_can)


def test_energy_partition_canonical():
    """Test energy partition for canonical vector"""
    interp = SeamInterpreter()
    v = np.array([1.0, 2.0, 3.0])
    v = v / np.linalg.norm(v)  # Normalize

    alpha_plus, alpha_minus = interp.energy_partition(v)

    # Should be mostly in even subspace
    assert alpha_plus > alpha_minus
    assert np.isclose(alpha_plus + alpha_minus, 1.0)


def test_energy_partition_anticanonical():
    """Test energy partition for anticanonical vector"""
    interp = SeamInterpreter()
    v = np.array([-1.0, -2.0, -3.0])
    v = v / np.linalg.norm(v)

    alpha_plus, alpha_minus = interp.energy_partition(v)

    # Should be mostly in odd subspace
    assert alpha_minus > alpha_plus
    assert np.isclose(alpha_plus + alpha_minus, 1.0)


def test_project_even():
    """Test even parity projection"""
    interp = SeamInterpreter()

    # Canonical vector
    v_can = np.array([1.0, 2.0, 3.0])
    v_can = v_can / np.linalg.norm(v_can)

    p_even = interp.project_even(v_can)

    # Should return same vector (already in even subspace)
    assert np.allclose(p_even, v_can)


def test_project_odd():
    """Test odd parity projection"""
    interp = SeamInterpreter()

    # Anticanonical vector
    v_anti = np.array([-1.0, -2.0, -3.0])
    v_anti = v_anti / np.linalg.norm(v_anti)

    p_odd = interp.project_odd(v_anti)

    # Should return flipped vector (canonical)
    v_can = -v_anti
    assert np.allclose(p_odd, v_can)


def test_execute_simple_load():
    """Test executing simple load"""
    v = np.array([1.0, 2.0, 3.0])
    interp = SeamInterpreter(variables={"q": v})

    ast = parse("q")
    result = interp.execute(ast)

    assert np.array_equal(result, v)


def test_execute_canonical():
    """Test executing canonical operation"""
    v = np.array([-1.0, 2.0, 3.0])
    interp = SeamInterpreter(variables={"q": v})

    ast = parse("q σ .")
    result = interp.execute(ast)

    # Result should have positive first element
    assert result[0] > 0


def test_orientation_tracking():
    """Test orientation tracking with seam operator"""
    from seam.types import Orientation

    interp = SeamInterpreter()
    assert interp.orientation == Orientation.POS

    # Parse and execute seam
    ast = parse("§")
    interp.execute(ast)

    # Orientation should flip
    assert interp.orientation == Orientation.NEG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
