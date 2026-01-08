"""
SEAM to AVDA Compiler

Translates SEAM programs to executable AVDA query plans.

Example:
    Input: q : α_± S_k* [P+ · k] [σ · k] ? top_k(10) .
    Output: Python/C++ code using ParityIndex
"""
import sys
sys.path.insert(0, '/home/user/SEAM-AVDA/src')

from seam.parser import parse, Load, Dup, Seam, ComputeEnergy, ProjectEven, ProjectOdd, Canonical, DotProduct, ConditionalSeam, TopK, Output
from seam.optimizer import optimize, cost_function
from typing import List, Dict


class CompilationError(Exception):
    """Raised when SEAM code cannot be compiled"""
    pass


def generate_python_plan(ast: List, variables: Dict[str, str]) -> str:
    """
    Generate Python code for SEAM program

    Args:
        ast: Optimized AST
        variables: Map of variable names to their sources (e.g., {"k": "database"})

    Returns:
        Python code as string
    """
    code = []
    code.append("import numpy as np")
    code.append("from libantipodal import ParityIndex")
    code.append("")
    code.append("def query_plan(query, database):")
    code.append("    \"\"\"Generated SEAM query plan\"\"\"")

    # Track stack with variable names
    stack_vars = []
    var_counter = [0]

    def new_var():
        var_counter[0] += 1
        return f"v{var_counter[0]}"

    for expr in ast:
        if isinstance(expr, Load):
            var = new_var()
            stack_vars.append(var)
            code.append(f"    {var} = {expr.var}")

        elif isinstance(expr, Dup):
            if not stack_vars:
                raise CompilationError("Stack underflow at dup")
            stack_vars.append(stack_vars[-1])

        elif isinstance(expr, ComputeEnergy):
            if not stack_vars:
                raise CompilationError("Stack underflow at α_±")
            v = stack_vars.pop()
            alpha_p = new_var()
            alpha_m = new_var()
            stack_vars.append(alpha_p)
            stack_vars.append(alpha_m)
            code.append(f"    # Compute energy partition")
            code.append(f"    from libantipodal import energy_partition")
            code.append(f"    {alpha_p}, {alpha_m} = energy_partition({v})")

        elif isinstance(expr, ConditionalSeam):
            # Generate conditional
            if len(stack_vars) < 2:
                raise CompilationError("Stack underflow at S_k*")

            alpha_m = stack_vars.pop()
            alpha_p = stack_vars.pop()

            code.append(f"    # Conditional seam S_k*")
            code.append(f"    alpha_max = max({alpha_p}, {alpha_m})")
            code.append(f"    K_STAR = {expr.threshold}")
            code.append(f"    if alpha_max > K_STAR:")

            # True branch (indented)
            saved_stack = stack_vars.copy()
            for e in expr.true_branch:
                # Recursively compile (simplified)
                if isinstance(e, ProjectEven):
                    v = stack_vars.pop()
                    result_var = new_var()
                    stack_vars.append(result_var)
                    code.append(f"        {result_var} = project_even(query)")
                elif isinstance(e, DotProduct):
                    v = stack_vars.pop()
                    result_var = new_var()
                    stack_vars.append(result_var)
                    code.append(f"        # Search parity index I+")
                    code.append(f"        {result_var} = database.search_parity_plus(query, k)")

            true_result = stack_vars[-1] if stack_vars else None

            # False branch
            code.append(f"    else:")
            stack_vars = saved_stack.copy()
            for e in expr.false_branch:
                if isinstance(e, Canonical):
                    v = stack_vars.pop() if stack_vars else "query"
                    result_var = new_var()
                    stack_vars.append(result_var)
                    code.append(f"        {result_var} = canonical(query)")
                elif isinstance(e, DotProduct):
                    v = stack_vars.pop()
                    result_var = new_var()
                    stack_vars.append(result_var)
                    code.append(f"        # Search quotient index I0")
                    code.append(f"        {result_var} = database.search_quotient(query, k)")

        elif isinstance(expr, TopK):
            if not stack_vars:
                raise CompilationError("Stack underflow at top_k")
            results = stack_vars.pop()
            output_var = new_var()
            stack_vars.append(output_var)
            code.append(f"    {output_var} = sorted({results}, key=lambda x: x[1], reverse=True)[:{expr.k}]")

        elif isinstance(expr, Output):
            if not stack_vars:
                raise CompilationError("Stack underflow at output")
            final_var = stack_vars.pop()
            code.append(f"    return {final_var}")

    # Add return if not explicit
    if stack_vars:
        code.append(f"    return {stack_vars[-1]}")

    return "\n".join(code)


def generate_cpp_plan(ast: List, variables: Dict[str, str]) -> str:
    """
    Generate C++ code for SEAM program

    Returns:
        C++ code as string
    """
    code = []
    code.append("// Generated by SEAM compiler")
    code.append("#include <antipodal/parity_index.h>")
    code.append("#include <vector>")
    code.append("")
    code.append("std::vector<SearchResult> query_plan(")
    code.append("    const Vector& query,")
    code.append("    ParityIndex& database,")
    code.append("    size_t k")
    code.append(") {")

    # Simple compilation (similar to Python but C++ syntax)
    code.append("    // Compute energy partition")
    code.append("    auto [alpha_plus, alpha_minus] = energy_partition(query);")
    code.append("    float alpha_max = std::max(alpha_plus, alpha_minus);")
    code.append("")
    code.append("    constexpr float K_STAR = 0.72134752044f;")
    code.append("")
    code.append("    if (alpha_max > K_STAR) {")
    code.append("        // Structure-dominated: use parity index")
    code.append("        Vector search_vec = (alpha_plus > alpha_minus)")
    code.append("            ? project_even(query)")
    code.append("            : project_odd(query);")
    code.append("        return database.search(search_vec, k);")
    code.append("    } else {")
    code.append("        // Entropy-dominated: use quotient index")
    code.append("        Vector search_vec = canonical(query);")
    code.append("        return database.search(search_vec, k);")
    code.append("    }")
    code.append("}")

    return "\n".join(code)


def compile_seam(
    seam_code: str,
    output_format: str = "python",
    optimize_code: bool = True,
    variables: Dict[str, str] = None
) -> str:
    """
    Compile SEAM code to executable format

    Args:
        seam_code: SEAM program as string
        output_format: "python" or "cpp"
        optimize_code: Whether to apply optimizations
        variables: Variable name mappings

    Returns:
        Generated code as string

    Example:
        >>> code = "q : α_± S_k* [P+ · k] [σ · k] ? top_k(10) ."
        >>> python_code = compile_seam(code, output_format="python")
    """
    variables = variables or {"q": "query", "k": "database"}

    # Parse
    ast = parse(seam_code)

    # Optimize
    if optimize_code:
        original_cost = cost_function(ast)
        ast = optimize(ast)
        optimized_cost = cost_function(ast)
        speedup = original_cost / optimized_cost if optimized_cost > 0 else 1.0
        print(f"Optimization: {original_cost:.2f} → {optimized_cost:.2f} (speedup: {speedup:.2f}×)")

    # Generate code
    if output_format == "python":
        return generate_python_plan(ast, variables)
    elif output_format == "cpp":
        return generate_cpp_plan(ast, variables)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def main():
    """CLI for SEAM compiler"""
    import argparse

    parser = argparse.ArgumentParser(description="SEAM to AVDA compiler")
    parser.add_argument("input", help="SEAM file or code string")
    parser.add_argument("--format", choices=["python", "cpp"], default="python",
                       help="Output format (default: python)")
    parser.add_argument("--no-optimize", action="store_true",
                       help="Disable optimization")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Read input
    try:
        with open(args.input, 'r') as f:
            seam_code = f.read()
    except FileNotFoundError:
        # Treat as code string
        seam_code = args.input

    # Compile
    generated_code = compile_seam(
        seam_code,
        output_format=args.format,
        optimize_code=not args.no_optimize
    )

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(generated_code)
        print(f"Generated code written to {args.output}")
    else:
        print(generated_code)


if __name__ == "__main__":
    main()
