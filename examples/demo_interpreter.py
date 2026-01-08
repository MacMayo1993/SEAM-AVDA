"""
SEAM Interpreter Demo

Demonstrates SEAM language execution with simple examples.
"""
import sys
sys.path.insert(0, '/home/user/SEAM-AVDA/src')

import numpy as np
from seam.parser import parse
from seam.interpreter import SeamInterpreter
from seam.types import TypeChecker


def example1_energy_partition():
    """Example 1: Compute energy partition α_±"""
    print("=" * 60)
    print("Example 1: Energy Partition")
    print("=" * 60)

    # Create a test vector
    v = np.array([1.0, 2.0, 3.0, -1.0])

    # SEAM code: q α_±
    seam_code = "q"

    # Parse and execute
    ast = parse(seam_code)
    interpreter = SeamInterpreter(variables={"q": v})

    # Type check
    type_checker = TypeChecker()
    final_state = type_checker.check(ast)
    print(f"Type check: {final_state}")

    # Execute
    result = interpreter.execute(ast)
    print(f"Input vector: {v}")
    print(f"Result: {result}")

    # Now compute energy
    from seam.interpreter import SeamInterpreter
    interp = SeamInterpreter(variables={"q": v})
    alpha_plus, alpha_minus = interp.energy_partition(v)
    print(f"α₊ = {alpha_plus:.4f}")
    print(f"α₋ = {alpha_minus:.4f}")
    print(f"α₊ + α₋ = {alpha_plus + alpha_minus:.4f}")
    print()


def example2_canonical():
    """Example 2: Canonical representative"""
    print("=" * 60)
    print("Example 2: Canonical Representative")
    print("=" * 60)

    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([-1.0, -2.0, -3.0])

    interp = SeamInterpreter()
    v1_can = interp.canonical(v1)
    v2_can = interp.canonical(v2)

    print(f"v1 = {v1}")
    print(f"σ(v1) = {v1_can}")
    print(f"v2 = {v2}")
    print(f"σ(v2) = {v2_can}")
    print(f"Are they equal? {np.allclose(v1_can, v2_can)}")
    print()


def example3_parity_projection():
    """Example 3: Parity projections P+, P-"""
    print("=" * 60)
    print("Example 3: Parity Projections")
    print("=" * 60)

    # Canonical vector
    v_can = np.array([1.0, 2.0, 3.0])
    v_can = v_can / np.linalg.norm(v_can)

    # Anticanonical vector
    v_anti = -v_can

    interp = SeamInterpreter()

    print(f"Canonical vector: {v_can}")
    print(f"P+(v_can) = {interp.project_even(v_can)}")
    print(f"P-(v_can) = {interp.project_odd(v_can)}")
    print()

    print(f"Anticanonical vector: {v_anti}")
    print(f"P+(v_anti) = {interp.project_even(v_anti)}")
    print(f"P-(v_anti) = {interp.project_odd(v_anti)}")
    print()


def example4_compile():
    """Example 4: Compile SEAM to Python"""
    print("=" * 60)
    print("Example 4: SEAM Compiler")
    print("=" * 60)

    from compiler.seam_to_avda import compile_seam

    # Simple SEAM query
    seam_code = "q σ ."

    print(f"SEAM code: {seam_code}")
    print()

    # Compile to Python
    python_code = compile_seam(seam_code, output_format="python", optimize_code=False)
    print("Generated Python code:")
    print("-" * 60)
    print(python_code)
    print()


if __name__ == "__main__":
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║         SEAM Language Demonstration                   ║")
    print("║  Esoteric Language for Non-Orientable Computing       ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()

    example1_energy_partition()
    example2_canonical()
    example3_parity_projection()
    example4_compile()

    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)
