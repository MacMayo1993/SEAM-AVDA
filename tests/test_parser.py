"""
Tests for SEAM parser
"""
import sys
sys.path.insert(0, '/home/user/SEAM-AVDA/src')

import pytest
from seam.parser import parse, Load, Seam, ComputeEnergy, ProjectEven, Canonical, DotProduct, TopK, Output, ParseError


def test_parse_simple_load():
    """Test parsing simple load operation"""
    ast = parse("q")
    assert len(ast) == 1
    assert isinstance(ast[0], Load)
    assert ast[0].var == "q"


def test_parse_seam_operator():
    """Test parsing seam operator §"""
    ast = parse("§")
    assert len(ast) == 1
    assert isinstance(ast[0], Seam)


def test_parse_energy():
    """Test parsing energy partition α_±"""
    ast = parse("α_±")
    assert len(ast) == 1
    assert isinstance(ast[0], ComputeEnergy)


def test_parse_canonical():
    """Test parsing canonical operator σ"""
    ast = parse("σ")
    assert len(ast) == 1
    assert isinstance(ast[0], Canonical)


def test_parse_sequence():
    """Test parsing sequence of operations"""
    ast = parse("q σ .")
    assert len(ast) == 3
    assert isinstance(ast[0], Load)
    assert isinstance(ast[1], Canonical)
    assert isinstance(ast[2], Output)


def test_parse_conditional():
    """Test parsing conditional seam S_k*"""
    ast = parse("S_k* [P+ · k] [σ · k] ?")
    assert len(ast) == 1
    from seam.parser import ConditionalSeam
    assert isinstance(ast[0], ConditionalSeam)
    assert len(ast[0].true_branch) == 2
    assert len(ast[0].false_branch) == 2


def test_parse_top_k():
    """Test parsing top_k operator"""
    ast = parse("top_k(10)")
    assert len(ast) == 1
    assert isinstance(ast[0], TopK)
    assert ast[0].k == 10


def test_parse_complete_query():
    """Test parsing complete SEAM query"""
    code = "q α_± S_k* [P+ · k] [σ · k] ? top_k(10) ."
    ast = parse(code)
    assert len(ast) == 4
    assert isinstance(ast[0], Load)
    assert isinstance(ast[1], ComputeEnergy)
    from seam.parser import ConditionalSeam
    assert isinstance(ast[2], ConditionalSeam)
    assert isinstance(ast[3], TopK)


def test_parse_with_comments():
    """Test parsing with comments"""
    code = """
    q % load query
    σ % canonicalize
    . % output
    """
    ast = parse(code)
    assert len(ast) == 3


def test_parse_error_invalid_syntax():
    """Test parse error on invalid syntax"""
    with pytest.raises(ParseError):
        parse("invalid @#$ syntax")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
