"""
SEAM Parser: Converts SEAM strings to AST nodes

Grammar (simplified):
  program := expr*
  expr := load | dup | seam | operator | conditional | type_annotation
  load := identifier
  seam := '§'
  operator := '·' | '+' | '-' | 'Φ'
  conditional := 'S' threshold '[' expr* ']' '[' expr* ']' '?'
"""
from dataclasses import dataclass
from typing import List, Union, Any
import re


# AST Node types
@dataclass
class Load:
    """Load variable onto stack"""
    var: str


@dataclass
class Dup:
    """Duplicate top of stack"""
    pass


@dataclass
class Seam:
    """Apply seam operator § (flip orientation, apply Φ to following ops)"""
    pass


@dataclass
class ComputeEnergy:
    """Compute α_± energy partition"""
    pass


@dataclass
class ProjectEven:
    """Project to even parity subspace (P+)"""
    pass


@dataclass
class ProjectOdd:
    """Project to odd parity subspace (P-)"""
    pass


@dataclass
class Canonical:
    """Canonicalize to quotient representative (σ)"""
    pass


@dataclass
class DotProduct:
    """Dot product with database"""
    database: str


@dataclass
class ConditionalSeam:
    """Conditional seam S_k* with threshold-based branching"""
    threshold: float
    true_branch: List['ASTNode']
    false_branch: List['ASTNode']


@dataclass
class TopK:
    """Select top-k results"""
    k: int


@dataclass
class Output:
    """Output result"""
    pass


@dataclass
class TypeAnnotation:
    """Type annotation (orientation : stack_type)"""
    orientation: str
    stack_type: str


# Union type for all AST nodes
ASTNode = Union[Load, Dup, Seam, ComputeEnergy, ProjectEven, ProjectOdd,
                Canonical, DotProduct, ConditionalSeam, TopK, Output, TypeAnnotation]


class ParseError(Exception):
    """Raised when SEAM code cannot be parsed"""
    pass


class SEAMParser:
    """Parser for SEAM language"""

    def __init__(self):
        self.tokens = []
        self.pos = 0

    def tokenize(self, code: str) -> List[str]:
        """Tokenize SEAM code"""
        # Remove comments (after %)
        code = re.sub(r'%.*', '', code)

        # Token patterns
        patterns = [
            (r'α_±', 'ALPHA_PM'),
            (r'S_k\*', 'COND_SEAM'),
            (r'P\+', 'PROJECT_EVEN'),
            (r'P-', 'PROJECT_ODD'),
            (r'top_k', 'TOP_K'),
            (r'σ', 'CANONICAL'),
            (r'§', 'SEAM'),
            (r'·', 'DOT'),
            (r'Φ', 'PHI'),
            (r'\[', 'LBRACKET'),
            (r'\]', 'RBRACKET'),
            (r'\?', 'QUESTION'),
            (r'\.', 'OUTPUT'),
            (r':', 'COLON'),
            (r'\(', 'LPAREN'),
            (r'\)', 'RPAREN'),
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENT'),
            (r'\d+\.?\d*', 'NUMBER'),
            (r'\s+', 'WHITESPACE'),
        ]

        tokens = []
        i = 0
        while i < len(code):
            matched = False
            for pattern, token_type in patterns:
                regex = re.compile(pattern)
                match = regex.match(code, i)
                if match:
                    value = match.group(0)
                    if token_type != 'WHITESPACE':  # Skip whitespace
                        tokens.append((token_type, value))
                    i = match.end()
                    matched = True
                    break

            if not matched:
                raise ParseError(f"Unexpected character at position {i}: {code[i]}")

        return tokens

    def peek(self) -> tuple:
        """Peek at current token without consuming"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (None, None)

    def consume(self, expected_type: str = None) -> tuple:
        """Consume current token"""
        if self.pos >= len(self.tokens):
            raise ParseError(f"Unexpected end of input, expected {expected_type}")

        token_type, value = self.tokens[self.pos]
        if expected_type and token_type != expected_type:
            raise ParseError(f"Expected {expected_type}, got {token_type}")

        self.pos += 1
        return (token_type, value)

    def parse_expr(self) -> ASTNode:
        """Parse single expression"""
        token_type, value = self.peek()

        if token_type == 'IDENT':
            self.consume()
            return Load(value)

        elif token_type == 'SEAM':
            self.consume()
            return Seam()

        elif token_type == 'ALPHA_PM':
            self.consume()
            return ComputeEnergy()

        elif token_type == 'PROJECT_EVEN':
            self.consume()
            return ProjectEven()

        elif token_type == 'PROJECT_ODD':
            self.consume()
            return ProjectOdd()

        elif token_type == 'CANONICAL':
            self.consume()
            return Canonical()

        elif token_type == 'DOT':
            self.consume()
            _, db_name = self.consume('IDENT')
            return DotProduct(db_name)

        elif token_type == 'COND_SEAM':
            self.consume()
            # Parse conditional: S_k* [true_branch] [false_branch] ?
            self.consume('LBRACKET')
            true_branch = []
            while self.peek()[0] != 'RBRACKET':
                true_branch.append(self.parse_expr())
            self.consume('RBRACKET')

            self.consume('LBRACKET')
            false_branch = []
            while self.peek()[0] != 'RBRACKET':
                false_branch.append(self.parse_expr())
            self.consume('RBRACKET')

            self.consume('QUESTION')

            # Default threshold k* ≈ 0.721
            return ConditionalSeam(0.72134752044, true_branch, false_branch)

        elif token_type == 'TOP_K':
            self.consume()
            self.consume('LPAREN')
            _, k_str = self.consume('NUMBER')
            self.consume('RPAREN')
            return TopK(int(k_str))

        elif token_type == 'OUTPUT':
            self.consume()
            return Output()

        elif token_type == 'COLON':
            # Type annotation
            self.consume()
            _, orientation = self.consume('IDENT')
            _, stack_type = self.consume('IDENT')
            return TypeAnnotation(orientation, stack_type)

        else:
            raise ParseError(f"Unexpected token: {token_type}")

    def parse(self, code: str) -> List[ASTNode]:
        """Parse SEAM code to AST"""
        self.tokens = self.tokenize(code)
        self.pos = 0

        ast = []
        while self.pos < len(self.tokens):
            ast.append(self.parse_expr())

        return ast


def parse(code: str) -> List[ASTNode]:
    """Parse SEAM code to AST (convenience function)"""
    parser = SEAMParser()
    return parser.parse(code)
