# SEAM Language Specification

**Version 0.1.0** | **Date: January 2026**

---

## 1. Introduction

SEAM (Stack-based Esoteric language for Antipodal Manifolds) is a domain-specific programming language designed for computations on quotient manifolds, particularly ℝPⁿ⁻¹ = ℂⁿ / ℤ₂.

### 1.1 Design Philosophy

- **Quotient-native**: Operations work directly on equivalence classes
- **Type-safe**: Z₂-graded type system prevents orientation errors
- **Optimizable**: Cost-based rewriting for performance
- **Executable math**: Programs are formal specifications

### 1.2 Key Features

- Stack-based execution model
- Orientation tracking (Z₂ symmetry)
- Seam operators for boundary handling
- Duality maps (Φ) for operation inversion
- Conditional branching based on structure

---

## 2. Syntax

### 2.1 Lexical Structure

**Tokens:**
```
IDENTIFIER  ::= [a-zA-Z_][a-zA-Z0-9_]*
NUMBER      ::= [0-9]+ ('.' [0-9]+)?
COMMENT     ::= '%' [^\n]*

OPERATORS   ::= '§' | 'σ' | 'α_±' | 'P+' | 'P-' | 'Φ'
              | 'S_k*' | 'top_k' | '·' | '$'

DELIMITERS  ::= '[' | ']' | '(' | ')' | '?' | '.' | ':'
```

**Whitespace:** Space, tab, newline (ignored except in strings)

### 2.2 Grammar

```ebnf
program     ::= statement* ;
statement   ::= expression | annotation ;
expression  ::= load | operator | conditional | output ;

load        ::= IDENTIFIER ;
operator    ::= seam | canonical | energy | project | dotprod | topk ;

seam        ::= '§' ;
canonical   ::= 'σ' ;
energy      ::= 'α_±' ;
project     ::= 'P+' | 'P-' ;
dotprod     ::= '·' IDENTIFIER ;
topk        ::= 'top_k' '(' NUMBER ')' ;

conditional ::= 'S' threshold '[' expression* ']' '[' expression* ']' '?' ;
threshold   ::= 'k*' | 'τ' | NUMBER ;

annotation  ::= ':' orientation stack_type ;
orientation ::= 'Pos' | 'Neg' ;
stack_type  ::= type_expr+ ;
type_expr   ::= 'Vec' '(' NUMBER ')' | 'Scalar' | 'Results' ;

output      ::= '.' ;
```

### 2.3 Example Programs

**Simple canonicalization:**
```seam
q σ .
```

**Energy partition:**
```seam
q α_± .
```

**Adaptive k-NN:**
```seam
q α_± S_k* [P+ · k] [σ · k] ? top_k(10) .
```

**With type annotation:**
```seam
q : Pos Vec(768)
σ : Pos Vec(768)
. : Pos
```

---

## 3. Semantics

### 3.1 Execution Model

SEAM programs execute on a **stack machine** with:
- **Value stack**: Holds vectors, scalars, results
- **Orientation state**: Tracks current Z₂ parity (Pos/Neg)
- **Variable environment**: Maps names to values

**Execution steps:**
1. Parse program to AST
2. Type-check using Z₂-graded rules
3. Execute each expression sequentially
4. Return final stack top

### 3.2 Operators

#### 3.2.1 Load (`q`)

**Signature:** `→ Vec(N)`
**Effect:** Push variable `q` onto stack

```seam
q  % Stack: [Vec(N)]
```

#### 3.2.2 Seam (`§`)

**Signature:** `Pos ↔ Neg`
**Effect:** Flip orientation, apply duality map to following operation

```seam
§  % Orientation: Pos → Neg
```

**Duality map Φ:**
- `Φ(·) = anti_dot` (antipodal inner product)
- `Φ(+) = -` (addition → subtraction)
- `Φ(encode) = decode`

#### 3.2.3 Canonical (`σ`)

**Signature:** `Vec(N) → Vec(N)[Can]`
**Effect:** Map vector to canonical quotient representative

**Algorithm:**
1. Normalize: `v_norm = v / ||v||`
2. Find first nonzero: `i = argmax(|v_norm|)`
3. If `v_norm[i] < 0`: flip `v_norm = -v_norm`
4. Return `v_norm`

```seam
q σ  % Stack: [Vec(N)[Can]]
```

**Property:** `σ(v) = σ(-v)` (quotient invariance)

#### 3.2.4 Energy Partition (`α_±`)

**Signature:** `Vec(N) → Scalar × Scalar`
**Effect:** Compute parity energy decomposition

**Algorithm:**
1. `v_can = σ(v)`
2. `alignment = ⟨v/||v||, v_can⟩`
3. If `alignment > 0`:
   - `α₊ = alignment²`
   - `α₋ = 1 - α₊`
4. Else:
   - `α₋ = alignment²`
   - `α₊ = 1 - α₋`

```seam
q α_±  % Stack: [Scalar(α₊), Scalar(α₋)]
```

**Property:** `α₊ + α₋ = 1`

#### 3.2.5 Parity Projections (`P+`, `P-`)

**Signature:** `Vec(N) → Vec(N)[Even/Odd]`
**Effect:** Project to even/odd parity subspace

**P+ (even):**
```python
def project_even(v):
    v_can = canonical(v)
    alignment = dot(v/norm(v), v_can)
    return v/norm(v) if alignment > 0 else zeros_like(v)
```

**P- (odd):**
```python
def project_odd(v):
    v_can = canonical(v)
    alignment = dot(v/norm(v), v_can)
    return -v/norm(v) if alignment < 0 else zeros_like(v)
```

```seam
q P+  % Stack: [Vec(N)[Even]]
q P-  % Stack: [Vec(N)[Odd]]
```

#### 3.2.6 Dot Product (`· k`)

**Signature:** `Vec(N) → Results`
**Effect:** Search database `k` with query vector

```seam
q · k  % Stack: [Results]
```

#### 3.2.7 Conditional Seam (`S_k*`)

**Signature:** `Scalar × Scalar → [branch]`
**Effect:** Branch based on threshold comparison

**Syntax:**
```seam
α_± S_k* [true_branch] [false_branch] ?
```

**Semantics:**
1. Pop `α₋`, `α₊` from stack
2. Compute `α_max = max(α₊, α₋)`
3. If `α_max > k*`: execute `true_branch`
4. Else: execute `false_branch`

**Threshold `k*`:**
- Default: `k* = 0.72134752044`
- Derived from equal-cost boundary: `2D + λs = D`

```seam
q α_± S_k* [P+ · k] [σ · k] ?
% If structured (α>k*): use parity index
% Else: use quotient index
```

#### 3.2.8 Top-K (`top_k(n)`)

**Signature:** `Results → Results`
**Effect:** Select top `n` results by score

```seam
results top_k(10)  % Stack: [Results[10]]
```

#### 3.2.9 Output (`.`)

**Signature:** `α → ()`
**Effect:** Output top of stack, end program

```seam
q σ .  % Output canonical representative
```

---

## 4. Type System

### 4.1 Z₂-Graded Types

**Base types:**
- `Vec(N)`: N-dimensional vector
- `Scalar`: Floating-point number
- `Results`: Search results (vector of IDs + scores)

**Type modifiers:**
- `[Can]`: Canonical representative
- `[Even]`, `[Odd]`: Parity subspace
- `[Seam]`: Seam-marked (boundary)

**Orientation:**
- `Pos`: Positive orientation (canonical)
- `Neg`: Negative orientation (flipped)

### 4.2 Type Rules

**Load:**
```
───────────────
Γ ⊢ q : Vec(N)
```

**Seam:**
```
Γ, Pos ⊢ e : τ
──────────────────
Γ, Neg ⊢ § e : Φ(τ)
```

**Canonical:**
```
Γ ⊢ v : Vec(N)
────────────────────────
Γ ⊢ σ v : Vec(N)[Can]
```

**Energy:**
```
Γ ⊢ v : Vec(N)
────────────────────────────────────
Γ ⊢ α_± v : Scalar × Scalar
```

**Conditional:**
```
Γ ⊢ e₁ : τ    Γ ⊢ e₂ : τ
──────────────────────────────────────
Γ ⊢ S_k* [e₁] [e₂] ? : τ
```

### 4.3 Type Checking Algorithm

```python
def type_check(expr, state):
    if isinstance(expr, Load):
        state.stack.push(Vec(N))

    elif isinstance(expr, Seam):
        state.orientation = flip(state.orientation)

    elif isinstance(expr, Canonical):
        v = state.stack.pop()
        assert v.is_vector()
        state.stack.push(Vec(N, Can))

    elif isinstance(expr, Energy):
        v = state.stack.pop()
        assert v.is_vector()
        state.stack.push(Scalar)
        state.stack.push(Scalar)

    # ... etc
```

---

## 5. Optimization

### 5.1 Cost Function

**Definition:**
```
C(P) = steps + λ|seams| + Σ|stack|
```

Where:
- `steps`: Number of operations
- `|seams|`: Count of seam operators `§`
- `|stack|`: Cumulative stack depth
- `λ`: Seam penalty (default: 0.5)

### 5.2 Rewrite Rules

**R1: Seam Idempotence**
```
§ § → id
```

**R2: Seam Commutation**
```
§ op → Φ(op) §
```

**R3: Canonical Idempotence**
```
σ σ → σ
```

**R4: Energy Caching**
```
α_± ... α_± → α_± dup ...
(if same vector)
```

**R5: Seam Hoisting**
```
S [§ e₁] [§ e₂] ? → § S [e₁] [e₂] ?
```

**R6: Dual Elimination**
```
§ σ § → σ
(if σ is self-dual)
```

### 5.3 Optimization Algorithm

**Greedy rewriting:**
```python
def optimize(ast):
    while True:
        improved = False
        for rule in RULES:
            new_ast = apply(rule, ast)
            if cost(new_ast) < cost(ast):
                ast = new_ast
                improved = True
                break
        if not improved:
            break
    return ast
```

---

## 6. Runtime Semantics

### 6.1 Stack Operations

**Push:**
```python
stack.push(value)
```

**Pop:**
```python
value = stack.pop()
if stack.empty():
    raise StackUnderflow()
```

**Peek:**
```python
value = stack.peek()  # Non-destructive
```

### 6.2 Orientation Tracking

**State:**
```python
class State:
    orientation: Orientation  # Pos or Neg
    stack: List[Value]
```

**Flip:**
```python
def seam():
    state.orientation = flip(state.orientation)
```

### 6.3 Error Handling

**Errors:**
- `StackUnderflow`: Pop from empty stack
- `TypeError`: Type mismatch
- `DimensionError`: Vector size mismatch
- `ParseError`: Invalid syntax

---

## 7. Standard Library

### 7.1 Predefined Constants

```seam
k* = 0.72134752044  % Phase boundary threshold
```

### 7.2 Common Patterns

**Quotient search:**
```seam
q σ · k top_k(10) .
```

**Parity search (even):**
```seam
q P+ · k_plus top_k(10) .
```

**Adaptive search:**
```seam
q α_± S_k* [P+ · k] [σ · k] ? top_k(10) .
```

**Negation search:**
```seam
q § · k top_k(10) .
```

---

## 8. Implementation Notes

### 8.1 Parser

- **Tool:** Hand-written recursive descent or pyparsing
- **Output:** AST with typed nodes
- **Comments:** `%` to end of line

### 8.2 Interpreter

- **Execution:** Stack machine with orientation state
- **Values:** NumPy arrays for vectors, floats for scalars

### 8.3 Compiler

- **Target:** Python, C++, LLVM IR
- **Optimization:** Apply rewrite rules before code generation

---

## 9. Future Extensions

### 9.1 ℤ₄ Quotients (Quaternionic)

```seam
% Four-valued orientation
q : Z4(0)  % 0, 1, 2, 3 (mod 4)
```

### 9.2 Differentiable SEAM

```seam
% Learnable seam placement
q S_learnable_k* [...] [...] ?
% Train k* via backprop
```

### 9.3 Parallel Seams

```seam
% Multiple seams in parallel
q || § e1 || § e2 || e3 .
```

---

## 10. Appendix

### 10.1 Complete Example

**Program:**
```seam
% Adaptive k-NN with compression
%
% Input: query vector q, database k
% Output: top-10 nearest neighbors
%
% Algorithm:
% 1. Compute energy partition α_±
% 2. If structured (α > k*):
%    - Project to parity subspace
%    - Search parity index
% 3. Else:
%    - Canonicalize to quotient
%    - Search quotient index
% 4. Return top-10

q : Pos Vec(768)         % Load query (BERT embedding)
α_± : Pos Scalar Scalar  % Compute energy

S_k* [                   % Conditional seam
    P+ : Pos Vec(768)[Even]  % True: structure-dominated
    · k : Pos Results
] [
    σ : Pos Vec(768)[Can]    % False: entropy-dominated
    · k : Pos Results
] ?

top_k(10) : Pos Results  % Select top-10
. : Pos                  % Output
```

**Execution trace:**
```
Step 1: Load q
  Stack: [Vec(768)]
  Orientation: Pos

Step 2: Compute α_±
  Stack: [0.85, 0.15]
  Orientation: Pos

Step 3: Conditional (α_max=0.85 > k*=0.721 → true branch)
  - Project even: Stack: [Vec(768)[Even]]
  - Dot product: Stack: [Results]

Step 4: Top-k
  Stack: [Results[10]]

Step 5: Output
  Return: Results[10]
```

---

**End of Specification**

Version 0.1.0 | January 2026
