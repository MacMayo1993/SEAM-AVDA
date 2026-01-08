"""
SEAM → AVDA Compiler

Compiles SEAM programs to executable Python/C++ code using AVDA indices.
"""

from .seam_to_avda import compile_seam, generate_python_plan, generate_cpp_plan

__all__ = ["compile_seam", "generate_python_plan", "generate_cpp_plan"]
