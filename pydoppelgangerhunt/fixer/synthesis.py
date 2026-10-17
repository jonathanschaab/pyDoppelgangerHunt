"""Shared helper function synthesis, parameter ordering, docstrings, and typing imports."""

from __future__ import annotations

import ast
import difflib
import re
import typing
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.config import normalize_path_string
from pydoppelgangerhunt.reporters import extract_unit_source_code
from pydoppelgangerhunt.fixer.binding import (
    _base_unit_name,
    _has_receiver_reference,
    _is_pair_in_method,
    _is_same_file_path,
    _populate_unit_receiver_metadata,
    _prune_unshared_receivers,
    _resolve_effective_binding,
)
from pydoppelgangerhunt.fixer._df_cache import _get_or_parse_tree
from pydoppelgangerhunt.fixer._df_visitors import _OuterExprVisitor
from pydoppelgangerhunt.fixer.dataflow import (
    _load_unit_file_text,
    has_async_generator_delegation_hazard,
    resolve_clone_pair_outputs,
)
from pydoppelgangerhunt.fixer.scope import (
    _determine_binding_kind,
    _normalize_receiver_attrs,
    _rank_param_kind,
    _unfold_receiver_attribute,
    analyze_unit_variable_scope,
    inspect_single_unit_scope,
    is_subroutine_unit,
)
from pydoppelgangerhunt.fixer.source import (
    _detect_indent_step,
    _extract_unit_body_lines,
    _find_multiline_string_continuation_lines,
    _slice_unit_token_lines,
    dedent_source_text,
    extract_unit_comments_and_pragmas,
    split_source_lines,
)

_ASSIGN_STMT_RE = re.compile(
    r"^(?:[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*|\[[^\]]+\])*|\w+(?:\s*,\s*\w+)+)"
    r"(?:\s*:\s*[^=]+)?\s*(?<![=<>!:])=(?!=)\s*(.+)$"
)


def _attach_pragmas(
    lines: List[str],
    pragmas: Sequence[Dict[str, Any]],
) -> List[str]:
    """Attaches extracted boundary pragmas (# type: ignore, # noqa) to lines if not present."""
    result = list(lines)
    for p in pragmas:
        if p.get("is_pragma"):
            p_text = str(p.get("text", "")).strip()
            if p_text and not any(p_text in ln for ln in result):
                for idx, ln in enumerate(result):
                    if ln.strip() and not ln.strip().startswith("#"):
                        result[idx] = ln.rstrip() + f"  {p_text}"
                        break
    return result


def _node_references_name(node: ast.AST, name: str) -> bool:
    """Checks whether an AST node contains any free load references to the identifier."""
    visitor = _OuterExprVisitor()
    visitor.visit(node)
    return name in visitor.names


def _parse_dedented_stmts(raw_text: str) -> List[ast.stmt]:
    """Parses a raw source block into AST statements, handling column-0 multiline strings."""
    dedented = dedent_source_text(raw_text)
    stripped = dedented.strip()
    if not stripped:
        return []
    try:
        return ast.parse(stripped).body
    except SyntaxError:
        try:
            wrapped = ast.parse(f"if 1:\n{dedented}").body
            if len(wrapped) == 1 and isinstance(wrapped[0], ast.If):
                return wrapped[0].body
        except SyntaxError:
            pass
    return []


def _strip_stmt_target_lines(lines: Sequence[str]) -> List[str]:
    """Strips assignment targets from statement lines to yield clean expression lines."""
    try:
        parsed = _parse_dedented_stmts("\n".join(lines))
        if len(parsed) > 1:
            return list(lines)
        if len(parsed) == 1:
            stmt = parsed[0]
            if isinstance(
                stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)
            ):
                if stmt.value is not None:
                    return ast.unparse(stmt.value).splitlines()
                return list(lines)
    except (SyntaxError, IndexError, ValueError):
        pass
    if len(lines) == 1:
        m = _ASSIGN_STMT_RE.match(lines[0].strip())
        if m:
            return [m.group(1).strip()]
    return list(lines)


def _extract_augassign_unreferenced_target(lines: Sequence[str]) -> Optional[str]:
    """Extracts target identifier of an AugAssign statement if not referenced in its RHS value."""
    try:
        parsed = _parse_dedented_stmts("\n".join(lines))
        if len(parsed) == 1 and isinstance(parsed[0], ast.AugAssign):
            if isinstance(parsed[0].target, ast.Name):
                target_name = parsed[0].target.id
                if not _node_references_name(parsed[0].value, target_name):
                    return target_name
    except (SyntaxError, IndexError, ValueError):
        pass
    return None


def _is_subline_unit(unit: Dict[str, Any]) -> bool:
    """Returns True if the unit represents a sub-line token span rather than a whole statement."""
    return any(
        unit.get(k) is not None
        for k in ("start_col", "end_col", "start_col_offset", "end_col_offset")
    )


def _align_pair_expr_kinds(u1: Dict[str, Any], u2: Dict[str, Any]) -> bool:
    """Aligns missing expression unit kinds across a clone pair or returns False on conflict."""
    k1_expr = u1.get("kind") in ("comprehension", "complex_expr")
    k2_expr = u2.get("kind") in ("comprehension", "complex_expr")
    if k1_expr != k2_expr:
        if u1.get("kind") and u2.get("kind"):
            return False
        eff_expr_kind = u1.get("kind") if k1_expr else u2.get("kind")
        u1.setdefault("kind", eff_expr_kind)
        u2.setdefault("kind", eff_expr_kind)
    return True


def _is_valid_expr_unit_lines(unit: Dict[str, Any], lines: Sequence[str]) -> bool:
    """Checks that a whole-line expression unit parses to a supported single-statement form."""
    if _is_subline_unit(unit):
        return True
    parsed = _parse_dedented_stmts("\n".join(lines))
    if len(parsed) > 1:
        return False
    if len(parsed) == 1:
        stmt = parsed[0]
        if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.Expr)):
            return True
        if isinstance(stmt, (ast.AnnAssign, ast.Return)) and stmt.value is not None:
            return True
        return False
    return True


def _indent_code_lines(lines: Sequence[str], prefix: str) -> List[str]:
    """Indents code lines by prefix while leaving multiline string continuation lines untouched."""
    cont_lines = _find_multiline_string_continuation_lines("\n".join(lines))
    out: List[str] = []
    for idx, ln in enumerate(lines, start=1):
        if idx in cont_lines:
            out.append(ln)
        elif not ln.strip():
            out.append("")
        elif ln.startswith("\x0c"):
            out.append(f"\x0c{prefix}{ln[1:]}")
        else:
            out.append(f"{prefix}{ln}")
    return out


def _extract_stmt_lhs_and_rhs_nodes(
    stmt: ast.AST,
) -> Tuple[List[ast.AST], Optional[ast.AST]]:
    """Extracts LHS target/annotation AST nodes and the RHS value node from a parsed AST statement."""
    if isinstance(stmt, ast.Assign) and stmt.value is not None:
        return list(stmt.targets), stmt.value
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        return [stmt.target, stmt.annotation], stmt.value
    if isinstance(stmt, ast.AugAssign):
        return [stmt.target], stmt.value
    if isinstance(stmt, ast.Return) and stmt.value is not None:
        return [], stmt.value
    if isinstance(stmt, ast.Expr):
        return [], stmt.value
    return [], None


def _extract_lhs_and_rhs_nodes(
    lines: Sequence[str],
) -> Tuple[List[ast.AST], Optional[ast.AST]]:
    """Extracts LHS target/annotation AST nodes and the RHS value node from an assignment or expression."""
    try:
        parsed = _parse_dedented_stmts("\n".join(lines))
        if len(parsed) == 1:
            return _extract_stmt_lhs_and_rhs_nodes(parsed[0])
    except (SyntaxError, IndexError, ValueError):
        pass
    return [], None


def _extract_unreferenced_lhs_loads(
    lines_or_stmt: Union[Sequence[str], ast.AST],
) -> Set[str]:
    """Extracts identifiers loaded in an assignment LHS or annotation but not referenced in RHS."""
    if isinstance(lines_or_stmt, ast.AST):
        lhs_nodes, rhs_node = _extract_stmt_lhs_and_rhs_nodes(lines_or_stmt)
    else:
        lhs_nodes, rhs_node = _extract_lhs_and_rhs_nodes(lines_or_stmt)
    if rhs_node is None or not lhs_nodes:
        return set()
    loaded_lhs = {
        child.id
        for lhs in lhs_nodes
        for child in ast.walk(lhs)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    return {
        name for name in loaded_lhs if not _node_references_name(rhs_node, name)
    }


def _order_inputs_by_rhs_loads(
    candidates: Sequence[str],
    rhs_node: ast.AST,
    receiver_names: Sequence[str] = ("self", "cls", "this", "klass"),
) -> List[str]:
    """Orders input identifiers by their first free-load appearance in rhs_node, keeping receivers first."""
    rhs_visitor = _OuterExprVisitor()
    rhs_visitor.visit(rhs_node)
    rhs_order = {name: idx for idx, name in enumerate(rhs_visitor.ordered_names)}
    fallback_base = len(rhs_order)
    rec_set = set(receiver_names)
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda pair: (
            0 if pair[1].lstrip("*") in rec_set else 1,
            rhs_order.get(pair[1].lstrip("*"), fallback_base + pair[0]),
            pair[0],
        )
    )
    return [item for _, item in indexed]


def _prune_unreferenced_lhs_scope(
    unit: Dict[str, Any],
    scope_dict: Dict[str, Any],
    lines: Sequence[str],
) -> Set[str]:
    """Prunes LHS-only loaded identifiers and receiver attributes for whole-line expr units."""
    if _is_subline_unit(unit) or unit.get("kind") not in (
        "comprehension",
        "complex_expr",
    ):
        return set()
    scope_dict["outputs"] = []
    unit.pop("outputs", None)
    unit.pop("precomputed_outputs", None)
    lhs_nodes, rhs_node = _extract_lhs_and_rhs_nodes(lines)
    if rhs_node is None:
        return set()
    rec_param = unit.get("receiver_param")
    rec_kind = unit.get("receiver_kind")
    inst_receivers = {"self", "this"}
    class_receivers = {"cls", "klass"}
    if rec_param:
        if rec_kind == "class":
            class_receivers.add(rec_param)
        elif rec_kind in ("instance", None):
            inst_receivers.add(rec_param)
    all_receivers = tuple(inst_receivers | class_receivers)

    unref_loads = _extract_unreferenced_lhs_loads(lines)
    aug_unref = _extract_augassign_unreferenced_target(lines)
    if aug_unref is not None:
        unref_loads.add(aug_unref)
    for key, rec_names in (("inputs", all_receivers), ("free_vars", ())):
        if key in scope_dict and isinstance(scope_dict[key], list):
            filtered = [x for x in scope_dict[key] if x not in unref_loads]
            scope_dict[key] = _order_inputs_by_rhs_loads(
                filtered, rhs_node, receiver_names=rec_names
            )
    if "inputs" in unit and isinstance(unit["inputs"], list):
        filtered_u = [x for x in unit["inputs"] if x not in unref_loads]
        unit["inputs"] = _order_inputs_by_rhs_loads(
            filtered_u, rhs_node, receiver_names=all_receivers
        )
    lhs_attrs = {
        unfolded
        for lhs in lhs_nodes
        for child in ast.walk(lhs)
        if isinstance(child, ast.Attribute)
        for unfolded in (
            _unfold_receiver_attribute(child, receiver_names=all_receivers),
        )
        if unfolded is not None
    }
    if not unref_loads and not lhs_attrs:
        return set()
    rhs_attrs = {
        unfolded
        for child in ast.walk(rhs_node)
        if isinstance(child, ast.Attribute)
        for unfolded in (
            _unfold_receiver_attribute(child, receiver_names=all_receivers),
        )
        if unfolded is not None
    }
    unref_attrs = lhs_attrs - rhs_attrs
    if "attrs_written" in scope_dict and lhs_attrs:
        scope_dict["attrs_written"] = [
            a for a in scope_dict["attrs_written"] if a not in lhs_attrs
        ]
    for attr_key in (
        "attrs_read",
        "attrs_written",
        "instance_attrs",
        "class_attrs",
    ):
        if attr_key in scope_dict:
            scope_dict[attr_key] = [
                a
                for a in scope_dict[attr_key]
                if a not in unref_attrs
                and not any(str(a).startswith(f"{rec}.") for rec in unref_loads)
            ]

    inputs_set = set(scope_dict.get("inputs", []))
    has_inst = bool(scope_dict.get("instance_attrs")) or bool(
        inputs_set & inst_receivers
    )
    has_cls = bool(scope_dict.get("class_attrs")) or bool(
        inputs_set & class_receivers
    )
    scope_dict["has_instance_binding"] = has_inst
    scope_dict["has_class_binding"] = has_cls
    scope_dict["has_receiver_access"] = has_inst or has_cls
    scope_dict["binding_kind"] = _determine_binding_kind(has_inst, has_cls)
    return unref_loads


def _prune_pair_lhs_and_augassign_scopes(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    lines1: Sequence[str],
    lines2: Sequence[str],
) -> bool:
    """Prunes unreferenced LHS loads and AugAssign targets, returning False on arity mismatch."""
    _prune_unreferenced_lhs_scope(u1, scope1, lines1)
    _prune_unreferenced_lhs_scope(u2, scope2, lines2)
    is_whole_expr1 = not _is_subline_unit(u1) and u1.get("kind") in (
        "comprehension",
        "complex_expr",
    )
    is_whole_expr2 = not _is_subline_unit(u2) and u2.get("kind") in (
        "comprehension",
        "complex_expr",
    )
    if not (is_whole_expr1 or is_whole_expr2):
        return True
    return len(scope1.get("inputs", [])) == len(scope2.get("inputs", []))


def _format_call_arguments(
    inputs: List[str],
    param_details: List[Dict[str, Any]],
    receiver_to_omit: Optional[str] = None,
    target_inputs: Optional[List[str]] = None,
    *,
    positional_target_map: bool = False,
) -> str:
    """Formats argument strings for helper call sites, preserving keyword-only, vararg, and kwarg syntax."""
    param_map: Dict[str, str] = {
        p["name"].lstrip("*"): p.get("kind", "pos") for p in param_details
    }
    sorted_inputs = sorted(inputs, key=lambda v: _rank_param_kind(v, param_map))
    if target_inputs is not None and len(target_inputs) == len(inputs):
        if not positional_target_map and set(inputs) == set(target_inputs):
            inp_to_target = {k: k for k in inputs}
        else:
            inp_to_target = dict(zip(inputs, target_inputs))
        sorted_targets = [inp_to_target.get(k, k) for k in sorted_inputs]
    else:
        sorted_targets = sorted_inputs

    formatted_args: List[str] = []
    for h_var, t_var in zip(sorted_inputs, sorted_targets):
        clean_h = h_var.lstrip("*")
        clean_t = t_var.lstrip("*")
        if receiver_to_omit and (
            clean_t == receiver_to_omit
            or clean_h == receiver_to_omit
            or (receiver_to_omit == "self" and clean_t in ("self", "this"))
            or (receiver_to_omit == "cls" and clean_t in ("cls", "klass"))
            or (receiver_to_omit == "receivers" and clean_t in ("self", "cls", "this", "klass"))
        ):
            continue
        kind = param_map.get(clean_h, "")
        if kind == "kwonly":
            formatted_args.append(f"{clean_h}={clean_t}")
        elif kind == "vararg" or (h_var.startswith("*") and not h_var.startswith("**")):
            formatted_args.append(f"*{clean_t}")
        elif kind == "kwarg" or h_var.startswith("**"):
            formatted_args.append(f"**{clean_t}")
        else:
            formatted_args.append(clean_t)
    return ", ".join(formatted_args)

def _advance_quote_state(
    char: str,
    active_quote: Optional[str],
    escaped: bool,
) -> Tuple[Optional[str], bool]:
    """Updates quote tracking state for a single character."""
    if active_quote:
        if escaped:
            return active_quote, False
        if char == "\\":
            return active_quote, True
        if char == active_quote:
            return None, False
        return active_quote, False
    if char in ("'", '"'):
        return char, False
    return None, False


_MATCHING_BRACKETS: Dict[str, str] = {"]": "[", ")": "(", "}": "{"}


def _split_delimited_type_string(
    text: str,
    delimiter: str,
    *,
    require_balanced: bool = True,
    allow_empty: bool = False,
) -> List[str]:
    """Splits a type string by a top-level delimiter, respecting brackets and quotes."""
    chunks: List[str] = []
    current: List[str] = []
    bracket_stack: List[str] = []
    active_quote: Optional[str] = None
    escaped = False
    has_imbalance = False
    for char in text:
        active_quote, escaped = _advance_quote_state(char, active_quote, escaped)
        if active_quote:
            current.append(char)
            continue
        if char in ("[", "(", "{"):
            bracket_stack.append(char)
        elif char in ("]", ")", "}"):
            if not bracket_stack or bracket_stack[-1] != _MATCHING_BRACKETS[char]:
                if require_balanced:
                    return []
                has_imbalance = True
            else:
                bracket_stack.pop()
        elif char == delimiter and not bracket_stack:
            item = "".join(current).strip()
            if item or allow_empty:
                chunks.append(item)
            current = []
            continue
        current.append(char)
    if bracket_stack or active_quote is not None or has_imbalance:
        if require_balanced:
            return []
        return [text.strip()] if text.strip() else []
    if current:
        item = "".join(current).strip()
        if item or allow_empty:
            chunks.append(item)
    return chunks


def _split_type_args(type_str: str) -> List[str]:
    """Splits top-level arguments of a generic type string, respecting brackets and quotes."""
    cleaned = type_str.strip()
    if not cleaned.endswith("]") or "[" not in cleaned:
        return []
    bracket_idx = cleaned.find("[")
    depth = 0
    active_quote: Optional[str] = None
    escaped = False
    for idx in range(bracket_idx, len(cleaned)):
        ch = cleaned[idx]
        active_quote, escaped = _advance_quote_state(ch, active_quote, escaped)
        if active_quote:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                if idx != len(cleaned) - 1:
                    return []
                break
            if depth < 0:
                return []
    if depth != 0 or active_quote is not None:
        return []
    inner = cleaned[bracket_idx + 1 : -1]
    return [
        c
        for c in _split_delimited_type_string(
            inner, ",", require_balanced=True, allow_empty=False
        )
        if c
    ]


def _split_pipe_union_args(type_str: str) -> List[str]:
    """Splits top-level pipe-separated types outside brackets and quotes."""
    return _split_delimited_type_string(
        type_str, "|", require_balanced=False, allow_empty=False
    )


def _flatten_union_args(args: Sequence[str]) -> List[str]:
    """Flattens nested Union, typing.Union, and Optional parameter types into a
    flat argument list."""
    flattened: List[str] = []
    for a in args:
        if (
            a.startswith(("Union[", "typing.Union[", "typing_extensions.Union["))
        ) and a.endswith("]"):
            flattened.extend(_split_type_args(a))
        elif (
            a.startswith(("Optional[", "typing.Optional[", "typing_extensions.Optional["))
        ) and a.endswith("]"):
            flattened.extend(_split_type_args(a))
            flattened.append("None")
        else:
            flattened.append(a)
    return flattened


def _format_union_or_optional(unique_parts: List[str]) -> str:
    """Formats unique union arguments, converting None unions to Optional."""
    if not unique_parts:
        return ""
    if any(p in ("None", "NoneType") for p in unique_parts):
        non_none = [p for p in unique_parts if p not in ("None", "NoneType")]
        if not non_none:
            return "None"
        if len(non_none) == 1:
            target = non_none[0]
            if (
                target.startswith(
                    ("Optional[", "typing.Optional[", "typing_extensions.Optional[")
                )
                and target.endswith("]")
            ):
                return target
            return f"Optional[{target}]"
        return f"Optional[Union[{', '.join(non_none)}]]"
    if len(unique_parts) == 1:
        return unique_parts[0]
    return f"Union[{', '.join(unique_parts)}]"


def _is_single_quoted_literal(text: str) -> bool:
    """Returns True if text is wrapped in a single pair of outer quotes closing at the end."""
    if len(text) < 2 or text[0] not in ("'", '"') or text[-1] != text[0]:
        return False
    active_quote: Optional[str] = None
    escaped = False
    for idx, ch in enumerate(text):
        active_quote, escaped = _advance_quote_state(ch, active_quote, escaped)
        if active_quote is None:
            return idx == len(text) - 1
    return False


def _normalize_pipe_unions(type_str: Optional[str]) -> str:
    """Normalizes PEP 604 pipe unions (A | B) to typing.Union[A, B] syntax.

    Recursively converts stringified annotations (PEP 563) while preserving literal
    contents inside Literal[...] and non-type metadata arguments inside Annotated[...].
    """
    if not isinstance(type_str, str):
        return "" if type_str is None else str(type_str)
    cleaned = type_str.strip()
    if not cleaned:
        return cleaned

    if _is_single_quoted_literal(cleaned):
        inner = cleaned[1:-1].strip()
        norm_inner = _normalize_pipe_unions(inner)
        return f"{cleaned[0]}{norm_inner}{cleaned[0]}"

    pipe_parts = _split_pipe_union_args(cleaned)
    if len(pipe_parts) > 1:
        norm_parts = [_normalize_pipe_unions(p) for p in pipe_parts]
        unique_parts = list(dict.fromkeys(_flatten_union_args(norm_parts)))
        return _format_union_or_optional(unique_parts)

    if "[" in cleaned and cleaned.endswith("]"):
        bracket_idx = cleaned.find("[")
        prefix = cleaned[:bracket_idx].strip()
        depth = 0
        active_quote: Optional[str] = None
        escaped = False
        closes_at_end = False
        for idx in range(bracket_idx, len(cleaned)):
            ch = cleaned[idx]
            active_quote, escaped = _advance_quote_state(ch, active_quote, escaped)
            if active_quote:
                continue
            if ch in ("[", "(", "{"):
                depth += 1
            elif ch in ("]", ")", "}"):
                depth -= 1
                if depth == 0:
                    closes_at_end = idx == len(cleaned) - 1
                    break
        if closes_at_end:
            if prefix in ("Literal", "typing.Literal", "typing_extensions.Literal"):
                return cleaned
            inner_args = _split_type_args(cleaned)
            if inner_args:
                if prefix in (
                    "Annotated",
                    "typing.Annotated",
                    "typing_extensions.Annotated",
                ):
                    norm_first = _normalize_pipe_unions(inner_args[0])
                    all_annotated = [norm_first] + list(inner_args[1:])
                    return f"{prefix}[{', '.join(all_annotated)}]"
                norm_args = [_normalize_pipe_unions(arg) for arg in inner_args]
                if prefix in ("Union", "typing.Union", "typing_extensions.Union"):
                    unique = list(dict.fromkeys(_flatten_union_args(norm_args)))
                    return _format_union_or_optional(unique)
                if prefix in (
                    "Optional",
                    "typing.Optional",
                    "typing_extensions.Optional",
                ):
                    unique = list(
                        dict.fromkeys(_flatten_union_args([*norm_args, "None"]))
                    )
                    return _format_union_or_optional(unique)
                return f"{prefix}[{', '.join(norm_args)}]"

    return cleaned


def _strip_container_module_prefix(type_str: str) -> str:
    """Strips leading typing/collections module prefixes from a type annotation."""
    for pfx in (
        "typing.",
        "typing_extensions.",
        "collections.abc.",
        "collections.",
    ):
        if type_str.startswith(pfx):
            return type_str[len(pfx) :]
    return type_str


def _unwrap_iterable_item_type(type_str: str) -> Optional[str]:
    """Unwraps the yielded element type from an iterable or union-of-iterables type."""
    norm_t = _normalize_pipe_unions(type_str)
    if not norm_t:
        return None

    if (
        norm_t.startswith("Union[") or norm_t.startswith("typing.Union[")
    ) and norm_t.endswith("]"):
        union_args = _split_type_args(norm_t)
        if not union_args:
            return None
        unwrapped_members: List[str] = []
        for arg in union_args:
            unwrapped = _unwrap_iterable_item_type(arg)
            if unwrapped is None:
                return None
            unwrapped_members.append(unwrapped)
        unique_types = list(dict.fromkeys(_flatten_union_args(unwrapped_members)))
        if len(unique_types) == 1:
            return unique_types[0]
        return f"Union[{', '.join(unique_types)}]"

    target_t = _strip_container_module_prefix(norm_t)
    prefixes = (
        "Iterator[", "Iterable[", "Generator[", "List[", "Sequence[", "Set[",
        "Tuple[", "Dict[", "Collection[", "Mapping[", "MutableMapping[",
        "DefaultDict[", "OrderedDict[", "list[", "set[", "tuple[", "dict[",
        "defaultdict[", "mapping[", "mutablemapping[", "sequence[", "iterable[",
        "iterator[", "collection[",
    )

    for prefix in prefixes:
        if target_t.startswith(prefix) and target_t.endswith("]"):
            parts = _split_type_args(target_t)
            if parts:
                if prefix.lower().startswith("tuple["):
                    if len(parts) == 2 and parts[1] == "...":
                        return _normalize_pipe_unions(parts[0])
                    unique_types = list(dict.fromkeys(
                        _normalize_pipe_unions(p) for p in parts
                    ))
                    if len(unique_types) == 1:
                        return unique_types[0]
                    return _normalize_pipe_unions(
                        f"Union[{', '.join(unique_types)}]"
                    )
                return _normalize_pipe_unions(parts[0])
            break
    return None


def _merge_types(t1: Optional[str], t2: Optional[str], strategy: str) -> str:
    """Merges two type annotations according to the specified merge strategy."""
    if t1 and t2:
        if t1 == t2:
            return _normalize_pipe_unions(t1)
        if strategy == "union":
            return _normalize_pipe_unions(f"Union[{t1}, {t2}]")
        return "Any"
    res = t1 or t2 or "Any"
    return _normalize_pipe_unions(res) if res != "Any" else "Any"


TYPING_SYMBOLS: Set[str] = set(typing.__all__)


def _extract_required_typing_imports(signature_or_func_text: str) -> List[str]:
    """Identifies typing symbols referenced in type annotations via AST inspection."""
    text = signature_or_func_text.strip()
    tree: Optional[ast.AST] = None
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        pass

    if tree is None and "->" in text:
        parts = text.split("->", 1)
        params_part = parts[0].strip()
        if params_part.startswith("(") and params_part.endswith(")"):
            params_part = params_part[1:-1].strip()
        ret_part = parts[1].strip()
        try:
            tree = ast.parse(f"def _sig_wrapper({params_part}) -> {ret_part}: pass")
        except (SyntaxError, ValueError, UnicodeDecodeError):
            pass

    if tree is None:
        try:
            tree = ast.parse(f"def _sig_wrapper({text}): pass")
        except (SyntaxError, ValueError, UnicodeDecodeError):
            pass

    if tree is None:
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
        return sorted(s for s in TYPING_SYMBOLS if s in tokens)

    has_defs = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign))
        for n in ast.walk(tree)
    )
    needed: Set[str] = set()
    for node in ast.walk(tree):
        annotations: List[ast.AST] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
            all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if node.args.vararg:
                all_args.append(node.args.vararg)
            if node.args.kwarg:
                all_args.append(node.args.kwarg)
            for a in all_args:
                if a.annotation is not None:
                    annotations.append(a.annotation)
        elif isinstance(node, ast.AnnAssign):
            if node.annotation is not None:
                annotations.append(node.annotation)
        elif not has_defs and isinstance(node, ast.Expr):
            annotations.append(node.value)

        for ann in annotations:
            for sub in ast.walk(ann):
                if isinstance(sub, ast.Name) and sub.id in TYPING_SYMBOLS:
                    needed.add(sub.id)
    return sorted(needed)


def _format_helper_parameters(
    inputs: Sequence[str],
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    *,
    effective_binding: str,
    is_static_clone: bool,
    is_class_receiver: bool,
    type_merge_strategy: str,
    inputs2: Optional[Sequence[str]] = None,
    receiver_param: Optional[str] = None,
    positional_target_map: bool = False,
) -> List[str]:
    """Formats helper function parameters with type annotations and default values."""
    params: List[str] = []
    seen_kwonly = False

    # Extract parameter descriptors
    descriptors: List[Dict[str, Any]] = []
    use_positional_meta2 = bool(
        inputs2 is not None
        and len(inputs2) == len(inputs)
        and (positional_target_map or set(inputs) != set(inputs2))
    )
    for idx, var in enumerate(inputs):
        m1 = meta1.get(var, {})
        if use_positional_meta2 and inputs2 is not None:
            m2 = meta2.get(inputs2[idx], {}) or meta2.get(var, {})
        else:
            m2 = meta2.get(var, {})
            if not m2 and inputs2 and idx < len(inputs2):
                m2 = meta2.get(inputs2[idx], {})

        t1 = m1.get("type")
        t2 = m2.get("type")
        resolved_type = _merge_types(t1, t2, type_merge_strategy)

        d1 = m1.get("default")
        d2 = m2.get("default")
        resolved_default = d1 if (d1 and d2 and d1 == d2) else None

        kind = m1.get("kind") or m2.get("kind") or "pos"
        descriptors.append({
            "var": var,
            "type": resolved_type,
            "default": resolved_default,
            "kind": kind,
        })

    # Validate and ensure canonical argument kind ordering (primary receiver -> secondary receiver -> pos -> vararg -> kwonly -> kwarg)
    primary_receiver = receiver_param or ("cls" if is_class_receiver else "self")
    secondary_receiver = "self" if is_class_receiver else "cls"
    all_receivers = {primary_receiver, "self", "cls", "this", "klass"}

    def _kind_rank(desc: Dict[str, Any]) -> Tuple[int, int]:
        var = desc["var"]
        kind = desc["kind"]
        if var == primary_receiver:
            return (0, 0)
        if var == secondary_receiver:
            return (0, 1)
        if var in all_receivers:
            return (0, 0)
        rank_map = {"pos": 1, "vararg": 2, "kwonly": 3, "kwarg": 4}
        return (rank_map.get(kind, 1), 2)

    descriptors.sort(key=_kind_rank)

    # In method binding mode, ensure receiver parameter exists (unless static)
    if (
        effective_binding == "method"
        and not is_static_clone
        and not any(desc["var"] in all_receivers for desc in descriptors)
    ):
        rec_var = primary_receiver
        descriptors.insert(0, {
            "var": rec_var,
            "type": "Any",
            "default": None,
            "kind": "pos",
        })

    # Validate positional argument default order (non-default cannot follow default)
    has_pos_default = False
    invalid_pos_defaults = False
    for desc in descriptors:
        if desc["kind"] == "pos":
            if desc["default"] is not None:
                has_pos_default = True
            elif has_pos_default:
                invalid_pos_defaults = True
                break
    if invalid_pos_defaults:
        for desc in descriptors:
            if desc["kind"] == "pos":
                desc["default"] = None

    # Render formatted parameters
    for desc in descriptors:
        var = desc["var"]
        resolved_type = (
            _normalize_pipe_unions(desc["type"]) if desc.get("type") else "Any"
        )
        resolved_default = desc["default"]
        kind = desc["kind"]
        if kind in ("vararg", "kwarg"):
            resolved_default = None
        prefix = "*" if kind == "vararg" else ("**" if kind == "kwarg" else "")
        var_name = f"{prefix}{var}"

        if (
            kind == "kwonly"
            and not seen_kwonly
            and not any(p.startswith("*") and not p.startswith("**") for p in params)
        ):
            params.append("*")
            seen_kwonly = True

        if var == primary_receiver and effective_binding == "method":
            params.append(var)
        elif resolved_default is not None:
            params.append(f"{var_name}: {resolved_type} = {resolved_default}")
        else:
            params.append(f"{var_name}: {resolved_type}")

    return params


def _wrap_conditional_optional(type_str: str, is_conditional: bool) -> str:
    """Wraps type annotation in Optional if conditionally assigned and None is absent."""
    opt_prefixes = ("Optional[", "typing.Optional[", "typing_extensions.Optional[")
    if (
        is_conditional
        and not type_str.startswith(opt_prefixes)
        and not re.search(r"\bNone\b", type_str)
    ):
        return f"Optional[{type_str}]"
    return type_str


def _infer_outputs_return_type(
    helper_outputs: List[str],
    conditional_outs: Set[str],
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    type_merge_strategy: str = "fallback_any",
    resolved_ret: str = "Any",
    outputs2: Optional[List[str]] = None,
) -> Optional[str]:
    """Infers the aggregated return type annotation for helper outputs."""
    if len(helper_outputs) >= 2:
        out_types: List[str] = []
        for idx, out_var in enumerate(helper_outputs):
            t1 = meta1.get(out_var, {}).get("type")
            if outputs2 and idx < len(outputs2):
                m2 = meta2.get(outputs2[idx], {})
            else:
                m2 = meta2.get(out_var, {})
            t2 = m2.get("type")
            t_merged = _merge_types(t1, t2, type_merge_strategy)
            u2_var = outputs2[idx] if (outputs2 and idx < len(outputs2)) else out_var
            is_conditional = (out_var in conditional_outs) or (u2_var in conditional_outs)
            t_merged = _wrap_conditional_optional(t_merged, is_conditional)
            out_types.append(_normalize_pipe_unions(t_merged))
        return f"Tuple[{', '.join(out_types)}]"

    if len(helper_outputs) == 1:
        out_var = helper_outputs[0]
        t1 = meta1.get(out_var, {}).get("type")
        if outputs2 and len(outputs2) >= 1:
            m2 = meta2.get(outputs2[0], {})
        else:
            m2 = meta2.get(out_var, {})
        t2 = m2.get("type")
        out_t = _merge_types(t1, t2, type_merge_strategy)
        fallback = resolved_ret
        unqualified_ret = _strip_container_module_prefix(resolved_ret)
        if unqualified_ret.startswith((
            "Generator[", "Iterator[", "Iterable[",
            "AsyncGenerator[", "AsyncIterator[", "AsyncIterable[",
        )):
            if unqualified_ret.startswith("Generator[") and unqualified_ret.endswith("]"):
                parts = _split_type_args(unqualified_ret)
                fallback = parts[2] if len(parts) >= 3 else "Any"
            else:
                fallback = "Any"
        return_type = out_t if out_t != "Any" else fallback
        u2_var = outputs2[0] if (outputs2 and len(outputs2) >= 1) else out_var
        is_conditional = (out_var in conditional_outs) or (u2_var in conditional_outs)
        return_type = _wrap_conditional_optional(return_type, is_conditional)
        return _normalize_pipe_unions(return_type)

    return None


def _infer_helper_return_type(
    resolved_ret: str,
    helper_outputs: List[str],
    conditional_outs: Set[str],
    scope: Dict[str, Any],
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    *,
    type_merge_strategy: str = "fallback_any",
    is_async: bool = False,
    is_subroutine: Optional[bool] = None,
    unit_kind: Optional[str] = None,
    outputs2: Optional[List[str]] = None,
) -> str:
    """Infers the return type annotation for a synthesized shared helper function."""
    resolved_ret = _normalize_pipe_unions(resolved_ret)
    if scope.get("has_yield"):
        unqualified_ret = _strip_container_module_prefix(resolved_ret)
        has_return_val = bool(helper_outputs or scope.get("has_return_value"))
        has_yield_assign = bool(scope.get("has_yield_assignment"))
        if not has_return_val and resolved_ret not in ("Any", "None"):
            if not is_async and unqualified_ret.startswith("Generator["):
                return resolved_ret
            if not has_yield_assign:
                if not is_async and unqualified_ret.startswith(("Iterator[", "Iterable[")):
                    return resolved_ret
                if is_async and unqualified_ret.startswith((
                    "AsyncGenerator[", "AsyncIterator[", "AsyncIterable[",
                )):
                    return resolved_ret
        if (
            not is_async
            and has_return_val
            and not helper_outputs
            and unqualified_ret.startswith("Generator[")
        ):
            return resolved_ret

        inferred_yield_type = None
        inferred_send_type = None
        inferred_gen_ret_type = None
        if (
            unqualified_ret.startswith((
                "Iterator[",
                "Iterable[",
                "Generator[",
                "AsyncIterator[",
                "AsyncIterable[",
                "AsyncGenerator[",
            ))
            and unqualified_ret.endswith("]")
        ):
            inner_parts = _split_type_args(unqualified_ret)
            if inner_parts:
                inferred_yield_type = _normalize_pipe_unions(inner_parts[0])
            if (
                unqualified_ret.startswith(("Generator[", "AsyncGenerator["))
                and len(inner_parts) >= 2
            ):
                if inner_parts[1] not in ("None", "Any"):
                    inferred_send_type = _normalize_pipe_unions(inner_parts[1])
            if unqualified_ret.startswith("Generator[") and len(inner_parts) >= 3:
                inferred_gen_ret_type = _normalize_pipe_unions(inner_parts[2])
        elif not has_return_val and resolved_ret not in ("Any", "None"):
            inferred_yield_type = _unwrap_iterable_item_type(resolved_ret)

        if not inferred_yield_type:
            for kind, name in scope.get("yield_expr_names", []):
                if name.startswith(":literal:"):
                    inferred_yield_type = _normalize_pipe_unions(name[len(":literal:") :])
                    break
                m_t = meta1.get(name, {}).get("type") or meta2.get(name, {}).get("type")
                if not m_t:
                    continue
                if kind == "yield":
                    inferred_yield_type = _normalize_pipe_unions(m_t)
                    break
                if kind == "yield_from":
                    inferred_yield_type = _unwrap_iterable_item_type(m_t)
                    if inferred_yield_type:
                        break

        yield_t = _normalize_pipe_unions(inferred_yield_type) if inferred_yield_type else "Any"
        if not is_async:
            outputs_ret = _infer_outputs_return_type(
                helper_outputs,
                conditional_outs,
                meta1,
                meta2,
                type_merge_strategy=type_merge_strategy,
                resolved_ret=resolved_ret,
                outputs2=outputs2,
            )
            is_iter_annotation = unqualified_ret.startswith((
                "Iterator[", "Iterable[", "Generator["
            ))
            fallback_ret = inferred_gen_ret_type or (
                _normalize_pipe_unions(resolved_ret)
                if resolved_ret not in ("Any", "None")
                and not is_iter_annotation
                and scope.get("has_return_value")
                else ("Any" if (scope.get("has_return_value") or helper_outputs) else None)
            )
            fallback_send = "object" if type_merge_strategy == "strict" else "Any"
            send_t = (
                _normalize_pipe_unions(inferred_send_type)
                if inferred_send_type
                else (fallback_send if scope.get("has_yield_assignment") else "None")
            )
            ret_t = outputs_ret or fallback_ret or ("None" if send_t != "None" else None)
            if ret_t is not None:
                return _normalize_pipe_unions(f"Generator[{yield_t}, {send_t}, {ret_t}]")
            return _normalize_pipe_unions(f"Iterator[{yield_t}]")

        fallback_send = "object" if type_merge_strategy == "strict" else "Any"
        send_t = (
            _normalize_pipe_unions(inferred_send_type)
            if inferred_send_type
            else (fallback_send if scope.get("has_yield_assignment") else "None")
        )
        if send_t != "None" or scope.get("has_yield_assignment"):
            return _normalize_pipe_unions(f"AsyncGenerator[{yield_t}, {send_t}]")
        return _normalize_pipe_unions(f"AsyncIterator[{yield_t}]")

    if is_subroutine is not None:
        is_sub = is_subroutine
    elif unit_kind is not None:
        is_sub = is_subroutine_unit({"kind": unit_kind})
    else:
        is_sub = False
    if not is_sub and resolved_ret not in ("Any", "None"):
        return _normalize_pipe_unions(resolved_ret)

    outputs_ret = _infer_outputs_return_type(
        helper_outputs,
        conditional_outs,
        meta1,
        meta2,
        type_merge_strategy=type_merge_strategy,
        resolved_ret=resolved_ret,
        outputs2=outputs2,
    )
    if outputs_ret is not None:
        return _normalize_pipe_unions(outputs_ret)

    if not helper_outputs and not scope.get("has_return"):
        if is_sub or resolved_ret == "Any":
            return "Any" if unit_kind in ("comprehension", "complex_expr") else "None"

    return _normalize_pipe_unions(resolved_ret)


def _format_helper_docstring(
    call_target: str,
    helper_outputs: List[str],
    scope: Dict[str, Any],
    *,
    is_async: bool = False,
    doc_indent: str = "    ",
    sub_indent: str = "        ",
    await_prefix: str = "",
) -> str:
    """Formats docstring with call-site example and control flow hazard warnings."""
    docstring_lines = [f"{doc_indent}\"\"\"Auto-extracted shared helper for duplicate logic."]
    is_async_gen = bool(scope.get("has_yield") and is_async)
    is_sync_gen = bool(scope.get("has_yield") and not is_async)
    if is_async_gen:
        call_site = f"async for _item in {call_target}(...):\n{sub_indent}yield _item"
    elif is_sync_gen:
        if helper_outputs:
            assign = ", ".join(helper_outputs) if len(helper_outputs) >= 2 else helper_outputs[0]
            call_site = f"{assign} = (yield from {call_target}(...))"
        else:
            call_site = f"yield from {call_target}(...)"
    elif helper_outputs:
        assign = ", ".join(helper_outputs) if len(helper_outputs) >= 2 else helper_outputs[0]
        call_site = f"{assign} = {await_prefix}{call_target}(...)"
    else:
        call_site = f"{await_prefix}{call_target}(...)"
    docstring_lines.append("")
    docstring_lines.append(f"{doc_indent}Call site:")
    docstring_lines.append(f"{sub_indent}{call_site}")

    hazards = scope.get("control_flow_hazards", [])
    if hazards:
        hazard_str = ", ".join(hazards)
        docstring_lines.append("")
        docstring_lines.append(
            f"{doc_indent}WARNING: Non-local control flow hazard detected ({hazard_str}). "
            "Direct extraction alters caller control flow semantics."
        )
    docstring_lines.append(f"{doc_indent}\"\"\"")
    return "\n".join(docstring_lines)


def synthesize_shared_helper_code(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    type_merge_strategy: str = "fallback_any",
    include_imports: bool = False,
    preserve_pragmas: bool = True,
    method_binding: str = "auto",
    indent: str = "",
    is_static: bool = False,
    receiver_kind: Optional[str] = None,
    step: Optional[str] = None,
    repo_root: Optional[str] = None,
    helper_name: Optional[str] = None,
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    skip_pre_unit_closures: bool = False,
    closure_strictness: Optional[str] = None,
    outputs: Optional[List[str]] = None,
    outputs2: Optional[List[str]] = None,
) -> str:
    """Synthesizes a proposed shared helper function stub from two clone units.

    Args:
        u1: First clone unit dictionary.
        u2: Second clone unit dictionary.
        type_merge_strategy: Strategy for merging conflicting types.
            Options: "fallback_any" (default, falls back to Any)
            or "union" (suggests typing.Union[T1, T2]).
        include_imports: Whether to prepend required typing and local imports.
        preserve_pragmas: Whether to preserve comments and pragmas (# type: ignore / # noqa)
            from the original clone units in the extracted helper function.
        method_binding: Scope binding strategy ("auto", "method", "module").
            "auto" emits a private class method if both clones reside in the same class,
            or a module-level helper with explicit receiver injection otherwise.
        indent: Base indentation prefix for helper definition lines (defaults to 4 spaces for methods).
        is_static: Whether the cloned methods are static methods.
        receiver_kind: Receiver kind of enclosing method ('class', 'instance', 'static').
        step: Indentation step per indentation level (defaults to "\t" for tabs, 4 spaces otherwise).
        repo_root: Optional root directory of the repository for relative path resolution.
        helper_name: Optional custom helper name override.
        skip_pre_unit_closures: Whether to bypass scanning pre-unit AST closures.
        closure_strictness: Closure isolation strictness ("strict" / "fail_closed" vs
            "lenient" / "fast" / "skip").
        outputs: Optional pre-resolved output variable names for u1.
        outputs2: Optional pre-resolved output variable names for u2.
    """
    u1 = dict(u1)
    u2 = dict(u2)
    if not _align_pair_expr_kinds(u1, u2):
        return ""
    lines1 = _slice_unit_token_lines(
        u1,
        _extract_unit_body_lines(u1, [ln.rstrip("\r\n") for ln in extract_unit_source_code(u1, repo_root=repo_root)]),
    )
    lines2 = _slice_unit_token_lines(
        u2,
        _extract_unit_body_lines(u2, [ln.rstrip("\r\n") for ln in extract_unit_source_code(u2, repo_root=repo_root)]),
    )

    base_name1 = _base_unit_name(u1)
    base_name2 = _base_unit_name(u2)
    if not helper_name:
        helper_name = (
            f"_shared_{base_name1}" if base_name1 == base_name2 else f"_shared_{base_name1}_{base_name2}"
        )

    orig_lines1 = list(lines1)
    orig_lines2 = list(lines2)
    pragmas1: List[Dict[str, Any]] = []
    pragmas2: List[Dict[str, Any]] = []
    if preserve_pragmas:
        raw1 = "\n".join(orig_lines1)
        raw2 = "\n".join(orig_lines2)
        pragmas1 = extract_unit_comments_and_pragmas(raw1, 1, len(orig_lines1))
        pragmas2 = extract_unit_comments_and_pragmas(raw2, 1, len(orig_lines2))

    is_expr_unit = u1.get("kind") in ("comprehension", "complex_expr")
    if is_expr_unit:
        if not (
            _is_valid_expr_unit_lines(u1, orig_lines1)
            and _is_valid_expr_unit_lines(u2, orig_lines2)
        ):
            return ""
        lines1 = _strip_stmt_target_lines(lines1)
        lines2 = _strip_stmt_target_lines(lines2)

    # Find common lines using difflib matching
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    common_lines: List[str] = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            common_lines.extend(lines1[i1:i2])

    if not common_lines or len(common_lines) < len(lines1):
        common_lines = lines1

    if preserve_pragmas:
        common_lines = _attach_pragmas(common_lines, pragmas1 + pragmas2)

    def _resolve_tree(
        given_tree: Optional[ast.AST],
        given_src: Optional[str],
        unit: Dict[str, Any],
    ) -> Optional[ast.AST]:
        if given_tree is not None:
            return given_tree
        raw_text = given_src or _load_unit_file_text(unit, repo_root=repo_root)
        if raw_text:
            return _get_or_parse_tree(raw_text)
        return None

    tree1 = _resolve_tree(tree1, source_text1, u1)
    tree2 = _resolve_tree(tree2, source_text2, u2)

    # Variable scope analysis for concrete parameter signatures.
    scopes: List[Dict[str, Any]] = []
    for u_item, s_text, t_item in (
        (u1, source_text1, tree1),
        (u2, source_text2, tree2),
    ):
        if s_text is not None:
            u_item.setdefault("source_text", s_text)
        _populate_unit_receiver_metadata(
            u_item, repo_root=repo_root, source_text=s_text, tree=t_item
        )
        s_item = inspect_single_unit_scope(u_item, repo_root=repo_root, tree=t_item)
        if u_item.get("enclosing_is_async") and s_item.get("has_yield"):
            s_item["is_async"] = True
            u_item["is_async"] = True
        scopes.append(s_item)
    scope1, scope2 = scopes[0], scopes[1]
    if not _prune_pair_lhs_and_augassign_scopes(
        u1, u2, scope1, scope2, orig_lines1, orig_lines2
    ):
        return ""
    scope = analyze_unit_variable_scope(
        u1,
        u2,
        repo_root=repo_root,
        tree1=tree1,
        tree2=tree2,
        scope1=scope1,
        scope2=scope2,
    )

    if scope1.get("is_async") or scope2.get("is_async"):
        scope["is_async"] = True

    enc1 = u1.get("enclosing_class")
    enc2 = u2.get("enclosing_class")
    f1 = normalize_path_string(str(u1.get("file") or ""))
    f2 = normalize_path_string(str(u2.get("file") or ""))
    enc1_start = u1.get("enclosing_class_start")
    enc2_start = u2.get("enclosing_class_start")
    is_same_class = bool(
        enc1
        and enc2
        and enc1 == enc2
        and _is_same_file_path(f1, f2, repo_root=repo_root)
        and (enc1_start is None or enc2_start is None or enc1_start == enc2_start)
    )

    inputs = list(scope["inputs"])

    def _unit_receiver_kind(u: Dict[str, Any], u_scope: Dict[str, Any]) -> str:
        if u.get("receiver_kind"):
            return str(u["receiver_kind"])
        if "receiver_kind" in u and u.get("receiver_kind") is None:
            return "none"
        if receiver_kind:
            return receiver_kind
        if u.get("kind") in ("closure", "class", "comprehension", "data_table", "complex_expr"):
            return "none"
        if u.get("is_static"):
            return "static"
        bk = u_scope.get("binding_kind")
        if bk in ("static", "class", "instance"):
            return str(bk)
        if is_static:
            return "static"
        if u.get("enclosing_class"):
            return "instance"
        return "none"

    k1 = _unit_receiver_kind(u1, scope1)
    k2 = _unit_receiver_kind(u2, scope2)
    receiver_kinds_differ = bool(k1 != k2)
    if receiver_kinds_differ and (
        _has_receiver_reference(u1, scope1, repo_root=repo_root)
        or _has_receiver_reference(u2, scope2, repo_root=repo_root)
    ):
        return ""

    if scope1.get("is_async") is None or scope2.get("is_async") is None:
        return ""
    if bool(scope1.get("is_async")) != bool(scope2.get("is_async")):
        return ""
    if bool(scope1.get("has_yield")) != bool(scope2.get("has_yield")):
        return ""
    if has_async_generator_delegation_hazard(scope1, scope2, scope):
        return ""
    if scope1.get("nonlocals") or scope2.get("nonlocals") or scope.get("nonlocals"):
        return ""
    if "type_param_closure_capture" in scope.get("control_flow_hazards", []):
        return ""
    if not is_same_class and not _is_same_file_path(f1, f2, repo_root=repo_root) and (
        scope1.get("globals") or scope2.get("globals") or scope.get("globals")
    ):
        return ""
    rec1 = u1.get("receiver_param") or ("cls" if k1 == "class" else "self")
    rec2 = u2.get("receiver_param") or ("cls" if k2 == "class" else "self")
    if (
        _normalize_receiver_attrs(scope1.get("attrs_read", []), rec1)
        != _normalize_receiver_attrs(scope2.get("attrs_read", []), rec2)
        or _normalize_receiver_attrs(scope1.get("attrs_written", []), rec1)
        != _normalize_receiver_attrs(scope2.get("attrs_written", []), rec2)
    ):
        return ""

    is_static_clone = bool(
        is_static
        or receiver_kind == "static"
        or (k1 == "static" and k2 == "static")
    )
    is_class_receiver = bool(
        receiver_kind == "class"
        or (k1 == "class" and k2 == "class")
    )

    has_super = bool(scope1.get("has_super") or scope2.get("has_super"))
    has_mangled = bool(scope1.get("has_mangled_names") or scope2.get("has_mangled_names"))
    is_in_method = _is_pair_in_method(
        u1,
        u2,
        method_binding=method_binding,
        has_super=has_super,
        has_mangled=has_mangled,
        has_method_context=bool(k1 != "none" and k2 != "none"),
    )

    effective_binding = _resolve_effective_binding(
        method_binding,
        is_same_class,
        is_static=is_static_clone,
        receiver_kinds_differ=receiver_kinds_differ,
        is_in_method=is_in_method,
        receiver_kind=k1,
    )

    if has_super and (effective_binding == "module" or is_static_clone):
        return ""

    if has_mangled and (effective_binding != "method" or not is_same_class or is_static_clone):
        return ""

    if (
        scope1.get("has_class_scoped_type_params")
        or scope2.get("has_class_scoped_type_params")
    ) and (effective_binding != "method" or not is_same_class):
        return ""

    if effective_binding == "module":
        inputs = _prune_unshared_receivers(inputs, u1, u2, scope1, scope2, repo_root=repo_root)

    if effective_binding == "method" and not indent:
        indent = "    "

    inputs2 = list(scope2.get("inputs", []))
    if effective_binding == "module":
        inputs2 = _prune_unshared_receivers(inputs2, u2, u1, scope2, scope1, repo_root=repo_root)

    meta1 = {p["name"].lstrip("*"): p for p in scope1.get("param_details", [])}
    meta2 = {p["name"].lstrip("*"): p for p in scope2.get("param_details", [])}

    rec_param = u1.get("receiver_param") or u2.get("receiver_param")
    params = _format_helper_parameters(
        inputs,
        meta1,
        meta2,
        effective_binding=effective_binding,
        is_static_clone=is_static_clone,
        is_class_receiver=is_class_receiver,
        type_merge_strategy=type_merge_strategy,
        inputs2=inputs2 if len(inputs2) == len(inputs) else None,
        receiver_param=rec_param,
        positional_target_map=is_expr_unit,
    )

    r1 = scope1.get("return_type")
    r2 = scope2.get("return_type")
    resolved_ret = _merge_types(r1, r2, type_merge_strategy)
    is_sub1 = is_subroutine_unit(u1, source_text=source_text1, tree=tree1)
    is_sub2 = is_subroutine_unit(u2, source_text=source_text2, tree=tree2)
    if is_sub1 != is_sub2:
        return ""
    is_sub = is_sub1

    eff_outputs: List[str]
    if outputs is not None and outputs2 is not None:
        eff_outputs, u2_outs = list(outputs), list(outputs2)
    else:
        resolved_outs = resolve_clone_pair_outputs(
            u1=u1,
            u2=u2,
            scope1=scope1,
            scope2=scope2,
            source_text1=source_text1,
            source_text2=source_text2,
            tree1=tree1,
            tree2=tree2,
            repo_root=repo_root,
            skip_pre_unit_closures=skip_pre_unit_closures,
            closure_strictness=closure_strictness,
            require_pairing=False,
        )
        if resolved_outs is None:
            return ""
        eff_outputs, u2_outs = resolved_outs

    conditional_outs = set(scope.get("conditional_outputs", []))
    is_async = bool(scope.get("is_async", False))
    func_keyword = "async def" if is_async else "def"
    await_prefix = "await " if is_async else ""

    # Variables altered via global or nonlocal statements must not be captured in the helper's return tuple
    g1 = set(scope.get("globals", []))
    nl1 = set(scope.get("nonlocals", []))
    g2 = set(scope2.get("globals", [])) if scope2 else set()
    nl2 = set(scope2.get("nonlocals", [])) if scope2 else set()
    if len(eff_outputs) == len(u2_outs):
        filtered_pairs = [
            (v1, v2)
            for v1, v2 in zip(eff_outputs, u2_outs)
            if v1 not in g1
            and v1 not in nl1
            and v2 not in g1
            and v2 not in nl1
            and v2 not in g2
            and v2 not in nl2
        ]
        helper_outputs = [p[0] for p in filtered_pairs]
        helper_outputs2 = [p[1] for p in filtered_pairs]
    else:
        helper_outputs = [v for v in eff_outputs if v not in g1 and v not in nl1]
        helper_outputs2 = [v for v in u2_outs if v not in g1 and v not in nl1]

    if has_async_generator_delegation_hazard(
        scope, scope1, scope2, has_outputs=bool(helper_outputs)
    ):
        return ""

    return_type = _infer_helper_return_type(
        resolved_ret,
        helper_outputs,
        conditional_outs,
        scope,
        meta1,
        meta2,
        type_merge_strategy=type_merge_strategy,
        is_async=is_async,
        is_subroutine=is_sub,
        unit_kind=u1.get("kind"),
        outputs2=helper_outputs2 if len(helper_outputs2) == len(helper_outputs) else None,
    )

    params_str = ", ".join(params) if params else "*args: Any, **kwargs: Any"

    # Dedent common_lines first so relative block indentation is normalized
    common_lines = [
        ln.rstrip("\r\n")
        for ln in split_source_lines(dedent_source_text("\n".join(common_lines)))
    ]

    # Prepend scope modifiers and conditional variable initializations to preserve runtime safety
    prefix_stmts: List[str] = []
    if scope.get("globals"):
        g_vars = sorted(set(scope["globals"]))
        if not any(ln.strip().startswith("global ") for ln in common_lines):
            prefix_stmts.append(f"global {', '.join(g_vars)}")
    for out_var in helper_outputs:
        if out_var in conditional_outs and out_var not in inputs:
            prefix_stmts.append(f"{out_var} = None")

    if prefix_stmts:
        common_lines = prefix_stmts + common_lines

    has_trailing_return = any(
        ln.strip() == "return"
        or (
            ln.strip().startswith("return")
            and len(ln.strip()) > 6
            and ln.strip()[6] in (" ", "\t", "(", "#")
        )
        for ln in common_lines[-3:]
    )
    if u1.get("kind") in ("comprehension", "complex_expr"):
        if not has_trailing_return:
            full_code = "\n".join(common_lines).strip()
            stripped_expr: Optional[str] = None
            try:
                parsed_body = _parse_dedented_stmts("\n".join(common_lines))
                if len(parsed_body) > 1:
                    return ""
                if len(parsed_body) == 1:
                    stmt_node = parsed_body[0]
                    if isinstance(stmt_node, ast.Assign):
                        stripped_expr = ast.unparse(stmt_node.value)
                    elif isinstance(stmt_node, ast.AnnAssign) and stmt_node.value is not None:
                        stripped_expr = ast.unparse(stmt_node.value)
                    elif isinstance(stmt_node, ast.AugAssign):
                        stripped_expr = ast.unparse(stmt_node.value)
                    elif isinstance(stmt_node, ast.Return):
                        stripped_expr = (
                            ast.unparse(stmt_node.value)
                            if stmt_node.value is not None
                            else "None"
                        )
                    elif isinstance(stmt_node, ast.Expr):
                        stripped_expr = full_code
                    else:
                        return ""
            except (SyntaxError, IndexError):
                pass

            if stripped_expr is not None:
                stripped_lines = stripped_expr.splitlines()
                if len(stripped_lines) == 1:
                    common_lines = [f"return {stripped_lines[0]}"]
                else:
                    eff_step = step or _detect_indent_step(indent)
                    expr_inner = _indent_code_lines(stripped_lines, eff_step)
                    common_lines = ["return ("] + expr_inner + [")"]
            elif len(common_lines) == 1:
                first_expr = common_lines[0].strip()
                m_assign = _ASSIGN_STMT_RE.match(first_expr)
                if m_assign:
                    first_expr = m_assign.group(1).strip()
                common_lines = [f"return {first_expr}"]
            else:
                eff_step = step or _detect_indent_step(indent)
                expr_inner = _indent_code_lines(common_lines, eff_step)
                common_lines = ["return ("] + expr_inner + [")"]
            if preserve_pragmas:
                common_lines = _attach_pragmas(common_lines, pragmas1 + pragmas2)
        helper_outputs = []
    elif (
        not has_trailing_return
        and helper_outputs
        and not (bool(scope.get("has_yield")) and is_async)
    ):
        if len(helper_outputs) >= 2:
            common_lines = common_lines + [f"return {', '.join(helper_outputs)}"]
        else:
            common_lines = common_lines + [f"return {helper_outputs[0]}"]

    if step is None:
        step = _detect_indent_step(indent)

    body_indent = indent + step
    doc_indent = indent + step
    sub_indent = indent + step + step
    if effective_binding == "method":
        rec_call = u1.get("receiver_param") or u2.get("receiver_param")
        if is_class_receiver:
            call_target = f"{rec_call or 'cls'}.{helper_name}"
        elif is_static_clone:
            call_target = f"__class__.{helper_name}"
        else:
            call_target = f"{rec_call or 'self'}.{helper_name}"
    else:
        call_target = helper_name

    docstring_str = _format_helper_docstring(
        call_target,
        helper_outputs,
        scope,
        is_async=is_async,
        doc_indent=doc_indent,
        sub_indent=sub_indent,
        await_prefix=await_prefix,
    )

    indented_body_parts = _indent_code_lines(common_lines, body_indent)
    indented_body = "\n".join(indented_body_parts)
    if effective_binding == "method":
        if is_class_receiver:
            dec_prefix = f"{indent}@classmethod\n"
        elif is_static_clone:
            dec_prefix = f"{indent}@staticmethod\n"
        else:
            dec_prefix = ""
    else:
        dec_prefix = ""
    tp_list = list(
        scope.get("type_params")
        or scope1.get("type_params")
        or scope2.get("type_params")
        or []
    )
    if tp_list and not is_sub:
        tp1_idents = {
            tp.split(":", 1)[0].split("=", 1)[0].lstrip("*").strip()
            for tp in (scope1.get("type_params") or [])
        }
        tp2_idents = {
            tp.split(":", 1)[0].split("=", 1)[0].lstrip("*").strip()
            for tp in (scope2.get("type_params") or [])
        }
        common_tp_idents = (
            (tp1_idents & tp2_idents)
            if (tp1_idents and tp2_idents)
            else (tp1_idents | tp2_idents)
        )
        ref_text = f"{params_str}\n{return_type}\n{indented_body}"
        kept_tps = [
            tp
            for tp in tp_list
            if (
                (ident := tp.split(":", 1)[0].split("=", 1)[0].lstrip("*").strip())
                in common_tp_idents
                or bool(re.search(rf"\b{re.escape(ident)}\b", ref_text))
            )
        ]
        while True:
            kept_tp_text = "\n".join([ref_text, *kept_tps])
            next_tps = [
                tp
                for tp in tp_list
                if (
                    (ident := tp.split(":", 1)[0].split("=", 1)[0].lstrip("*").strip())
                    in common_tp_idents
                    or bool(re.search(rf"\b{re.escape(ident)}\b", kept_tp_text))
                )
            ]
            if len(next_tps) == len(kept_tps):
                tp_list = next_tps
                break
            kept_tps = next_tps
    type_params_str = f"[{', '.join(tp_list)}]" if (tp_list and not is_sub) else ""
    helper_def = (
        f"{dec_prefix}"
        f"{indent}{func_keyword} {helper_name}{type_params_str}({params_str}) -> {return_type}:\n"
        f"{docstring_str}\n"
        f"{indented_body}\n"
    )

    hazards = scope.get("control_flow_hazards", [])
    has_naked_ctrl = any(h in ("naked_break", "naked_continue") for h in hazards)
    if not has_naked_ctrl:
        try:
            if indent:
                compile("if 1:\n" + helper_def, "<synthesized_helper>", "exec")
            else:
                compile(helper_def, "<synthesized_helper>", "exec")
        except (SyntaxError, ValueError):
            return ""

    if include_imports:
        needed_typing = _extract_required_typing_imports(helper_def)
        import_header_parts: List[str] = []
        if needed_typing:
            import_header_parts.append(f"from typing import {', '.join(sorted(needed_typing))}")
        for loc_imp in scope.get("local_imports", []):
            if loc_imp not in import_header_parts:
                import_header_parts.append(loc_imp)
        if import_header_parts:
            combined = "\n".join(import_header_parts) + "\n\n\n" + helper_def
            try:
                compile("\n".join(import_header_parts), "<synthesized_imports>", "exec")
            except (SyntaxError, ValueError):
                return ""
            return combined

    return helper_def
