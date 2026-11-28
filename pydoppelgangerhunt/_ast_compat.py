"""Cross-version AST node type tuples and predicates for Python 3.9+ compatibility.

# INVARIANT (SINGLE AST COMPATIBILITY LAYER):
# All Python 3.10+/3.11+/3.12+ optional AST node checks (``Match``, ``TryStar``,
# ``TypeAlias``, PEP 695 type parameters) MUST route through this module instead of
# scattering ``getattr(ast, ...)`` or ``type(node).__name__ == ...`` checks across
# parsers, dependency graphs, and dataflow analyzers.
"""

from __future__ import annotations

import ast
from typing import Tuple, Type

from pydoppelgangerhunt.source_lines import (
    is_cpython_parser_stack_overflow,
    safe_parse_ast,
)

__all__ = [
    "AST_MATCH_AS_TYPES",
    "AST_MATCH_CASE_TYPES",
    "AST_MATCH_CLASS_TYPES",
    "AST_MATCH_MAPPING_TYPES",
    "AST_MATCH_OR_TYPES",
    "AST_MATCH_STAR_TYPES",
    "AST_MATCH_TYPES",
    "AST_TRYSTAR_TYPES",
    "AST_TRY_OR_TRYSTAR_TYPES",
    "AST_TYPE_ALIAS_TYPES",
    "AST_TYPE_PARAM_TYPES",
    "is_cpython_parser_stack_overflow",
    "is_match_as_node",
    "is_match_case_node",
    "is_match_class_node",
    "is_match_mapping_node",
    "is_match_node",
    "is_match_or_node",
    "is_match_star_node",
    "is_try_or_trystar_node",
    "is_trystar_node",
    "is_type_alias_node",
    "is_type_param_node",
    "safe_parse_ast",
]

AST_MATCH_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.Match,) if hasattr(ast, "Match") else ()
)
AST_MATCH_CASE_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.match_case,) if hasattr(ast, "match_case") else ()
)
AST_MATCH_AS_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.MatchAs,) if hasattr(ast, "MatchAs") else ()
)
AST_MATCH_STAR_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.MatchStar,) if hasattr(ast, "MatchStar") else ()
)
AST_MATCH_MAPPING_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.MatchMapping,) if hasattr(ast, "MatchMapping") else ()
)
AST_MATCH_CLASS_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.MatchClass,) if hasattr(ast, "MatchClass") else ()
)
AST_MATCH_OR_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.MatchOr,) if hasattr(ast, "MatchOr") else ()
)
AST_TRYSTAR_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.TryStar,) if hasattr(ast, "TryStar") else ()
)
AST_TRY_OR_TRYSTAR_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.Try, *AST_TRYSTAR_TYPES)
)
AST_TYPE_ALIAS_TYPES: Tuple[Type[ast.AST], ...] = (
    (ast.TypeAlias,) if hasattr(ast, "TypeAlias") else ()
)
AST_TYPE_PARAM_TYPES: Tuple[Type[ast.AST], ...] = tuple(
    getattr(ast, name)
    for name in ("TypeVar", "ParamSpec", "TypeVarTuple")
    if hasattr(ast, name)
)


def is_match_node(node: object) -> bool:
    """Returns True when *node* is an ast.Match statement (Python 3.10+)."""
    return bool(AST_MATCH_TYPES and isinstance(node, AST_MATCH_TYPES))


def is_match_case_node(node: object) -> bool:
    """Returns True when *node* is an ast.match_case clause (Python 3.10+)."""
    return bool(AST_MATCH_CASE_TYPES and isinstance(node, AST_MATCH_CASE_TYPES))


def is_match_as_node(node: object) -> bool:
    """Returns True when *node* is an ast.MatchAs pattern (Python 3.10+)."""
    return bool(AST_MATCH_AS_TYPES and isinstance(node, AST_MATCH_AS_TYPES))


def is_match_star_node(node: object) -> bool:
    """Returns True when *node* is an ast.MatchStar pattern (Python 3.10+)."""
    return bool(AST_MATCH_STAR_TYPES and isinstance(node, AST_MATCH_STAR_TYPES))


def is_match_mapping_node(node: object) -> bool:
    """Returns True when *node* is an ast.MatchMapping pattern (Python 3.10+)."""
    return bool(
        AST_MATCH_MAPPING_TYPES and isinstance(node, AST_MATCH_MAPPING_TYPES)
    )


def is_match_class_node(node: object) -> bool:
    """Returns True when *node* is an ast.MatchClass pattern (Python 3.10+)."""
    return bool(AST_MATCH_CLASS_TYPES and isinstance(node, AST_MATCH_CLASS_TYPES))


def is_match_or_node(node: object) -> bool:
    """Returns True when *node* is an ast.MatchOr pattern (Python 3.10+)."""
    return bool(AST_MATCH_OR_TYPES and isinstance(node, AST_MATCH_OR_TYPES))


def is_trystar_node(node: object) -> bool:
    """Returns True when *node* is an ast.TryStar statement (Python 3.11+)."""
    return bool(AST_TRYSTAR_TYPES and isinstance(node, AST_TRYSTAR_TYPES))


def is_try_or_trystar_node(node: object) -> bool:
    """Returns True when *node* is an ast.Try or ast.TryStar statement."""
    return isinstance(node, AST_TRY_OR_TRYSTAR_TYPES)


def is_type_alias_node(node: object) -> bool:
    """Returns True when *node* is an ast.TypeAlias statement (Python 3.12+)."""
    return bool(AST_TYPE_ALIAS_TYPES and isinstance(node, AST_TYPE_ALIAS_TYPES))


def is_type_param_node(node: object) -> bool:
    """Returns True when *node* is a PEP 695 type_param AST node (Python 3.12+)."""
    return bool(AST_TYPE_PARAM_TYPES and isinstance(node, AST_TYPE_PARAM_TYPES))
