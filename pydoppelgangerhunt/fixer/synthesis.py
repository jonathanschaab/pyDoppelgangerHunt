"""Shared helper function synthesis, parameter ordering, docstrings, and typing imports."""

from __future__ import annotations

import ast
import copy
import difflib
import re
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.config import (
    normalize_path_string,
    resolve_pair_closure_policy,
)
from pydoppelgangerhunt.fixer._pair_eval import (
    check_pair_core_hazards,
    evaluate_clone_pair_scopes,
)
from pydoppelgangerhunt.fixer._synthesis_types import (  # pylint: disable=unused-import,useless-import-alias
    TYPING_SYMBOLS as TYPING_SYMBOLS,
    _extract_required_typing_imports,
    _format_helper_docstring,
    _format_helper_parameters,
    _infer_helper_return_type,
    _merge_types,
)
from pydoppelgangerhunt.fixer.binding import (
    _base_unit_name,
    _has_receiver_reference,
    _is_pair_in_method,
    _is_same_file_path,
    _populate_unit_receiver_metadata,
    _prune_unshared_receivers,
    _resolve_effective_binding,
)
from pydoppelgangerhunt._ast_compat import (
    is_cpython_parser_stack_overflow,
    safe_parse_ast,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_parse_tree,
    _with_analysis_session,
)
from pydoppelgangerhunt.fixer._df_visitors import _OuterExprVisitor
from pydoppelgangerhunt.fixer.dataflow import (
    allows_u2_source_placeholder,
    has_async_generator_delegation_hazard,
    resolve_clone_pair_outputs,
)
from pydoppelgangerhunt.fixer.source import _load_unit_file_text
from pydoppelgangerhunt.fixer.scope import (
    _determine_binding_kind,
    _normalize_receiver_attrs,
    _rank_param_kind,
    _unfold_receiver_attribute,
    analyze_unit_variable_scope,
    is_subroutine_unit,
    resolve_unit_inferred_col_bounds,
    scope_matches_closure_mode,
)
from pydoppelgangerhunt.fixer.source import (
    _detect_indent_step,
    _extract_unit_token_lines_with_state,
    _find_multiline_string_continuation_lines,
    validate_unit_schema,
    dedent_source_text,
    extract_unit_comments_and_pragmas,
    split_source_lines,
)
from pydoppelgangerhunt.source_lines import unit_file_str
from pydoppelgangerhunt.unit_boundary import END_COL_KEYS, START_COL_KEYS

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
    try:
        visitor = _OuterExprVisitor()
        visitor.visit(node)
        return name in visitor.names
    except RecursionError:
        return True


def _parse_dedented_stmts(raw_text: str) -> List[ast.stmt]:
    """Parses a raw source block into AST statements, handling column-0 multiline strings."""
    dedented = dedent_source_text(raw_text)
    stripped = dedented.strip()
    if not stripped:
        return []
    parsed_mod = safe_parse_ast(stripped)
    if parsed_mod is not None:
        return parsed_mod.body
    wrapped_mod = safe_parse_ast(f"if 1:\n{dedented}")
    if (
        wrapped_mod is not None
        and len(wrapped_mod.body) == 1
        and isinstance(wrapped_mod.body[0], ast.If)
    ):
        return wrapped_mod.body[0].body
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
                    return [
                        ln.rstrip("\r\n")
                        for ln in split_source_lines(ast.unparse(stmt.value))
                    ]
                return list(lines)
    except (SyntaxError, IndexError, ValueError, RecursionError):
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
    except (SyntaxError, IndexError, ValueError, RecursionError):
        pass
    return None


def _is_subline_unit(unit: Dict[str, Any]) -> bool:
    """Returns True if the unit represents a sub-line token span rather than a whole statement."""
    return any(unit.get(k) is not None for k in (*START_COL_KEYS, *END_COL_KEYS))


def _align_pair_expr_kinds(u1: Dict[str, Any], u2: Dict[str, Any]) -> bool:
    """Aligns missing expression unit kinds across a clone pair or returns False on conflict."""
    k1_expr = u1.get("kind") in ("comprehension", "complex_expr")
    k2_expr = u2.get("kind") in ("comprehension", "complex_expr")
    if k1_expr != k2_expr:
        if u1.get("kind") and u2.get("kind"):
            return False
        eff_expr_kind = u1.get("kind") if k1_expr else u2.get("kind")
        if not u1.get("kind"):
            u1["kind"] = eff_expr_kind
        if not u2.get("kind"):
            u2["kind"] = eff_expr_kind
    return True


def _is_valid_expr_unit_lines(unit: Dict[str, Any], lines: Sequence[str]) -> bool:
    """Checks that a whole-line expression unit parses to a supported single-statement form."""
    if _is_subline_unit(unit):
        return True
    parsed = _parse_dedented_stmts("\n".join(lines))
    if len(parsed) != 1:
        return False
    stmt = parsed[0]
    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.Expr)):
        return True
    if isinstance(stmt, (ast.AnnAssign, ast.Return)) and stmt.value is not None:
        return True
    return False


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
    except (SyntaxError, IndexError, ValueError, RecursionError):
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
    try:
        lhs_visitor = _OuterExprVisitor()
        for lhs in lhs_nodes:
            lhs_visitor.visit(lhs)
        return {
            name for name in lhs_visitor.names if not _node_references_name(rhs_node, name)
        }
    except RecursionError:
        return set()


def _order_inputs_by_rhs_loads(
    candidates: Sequence[str],
    rhs_node: ast.AST,
    receiver_names: Sequence[str] = ("self", "cls", "this", "klass"),
) -> List[str]:
    """Orders input identifiers by their first free-load appearance in rhs_node, keeping receivers first."""
    try:
        rhs_visitor = _OuterExprVisitor()
        rhs_visitor.visit(rhs_node)
        ordered = rhs_visitor.ordered_names
    except RecursionError:
        ordered = []
    rhs_order = {name: idx for idx, name in enumerate(ordered)}
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
    has_implicit_receiver = bool(scope_dict.get("has_super")) or bool(
        scope_dict.get("has_mangled_names")
    )
    if has_implicit_receiver:
        unref_loads -= set(all_receivers)
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
    rhs_attr_pairs = [
        (unfolded, isinstance(child.ctx, (ast.Store, ast.Del)))
        for child in ast.walk(rhs_node)
        if isinstance(child, ast.Attribute)
        for unfolded in (
            _unfold_receiver_attribute(child, receiver_names=all_receivers),
        )
        if unfolded is not None
    ]
    rhs_attrs = {unfolded for unfolded, _ in rhs_attr_pairs}
    rhs_written_attrs = {
        unfolded for unfolded, is_write in rhs_attr_pairs if is_write
    }
    unref_attrs = lhs_attrs - rhs_attrs
    lhs_only_written = lhs_attrs - rhs_written_attrs
    if "attrs_written" in scope_dict and lhs_only_written:
        scope_dict["attrs_written"] = [
            a for a in scope_dict["attrs_written"] if a not in lhs_only_written
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
    has_inst = (
        bool(scope_dict.get("instance_attrs"))
        or bool(inputs_set & inst_receivers)
        or (has_implicit_receiver and bool(scope_dict.get("has_instance_binding")))
    )
    has_cls = (
        bool(scope_dict.get("class_attrs"))
        or bool(inputs_set & class_receivers)
        or (has_implicit_receiver and bool(scope_dict.get("has_class_binding")))
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
    """Prunes unreferenced LHS loads and AugAssign targets, returning False on arity or receiver mismatch."""
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
    if (
        bool(scope1.get("has_instance_binding"))
        != bool(scope2.get("has_instance_binding"))
        or bool(scope1.get("has_class_binding"))
        != bool(scope2.get("has_class_binding"))
    ):
        return False
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


def _extract_probe_active_idents(probe_src: str, return_type: str) -> Set[str]:
    """Extracts referenced AST Name identifiers and annotation string-constant identifiers."""
    probe_tree = safe_parse_ast(probe_src)
    if probe_tree is None:
        return set(re.findall(r"\b[A-Za-z_]\w*\b", f"{probe_src}\n{return_type}"))
    all_nodes = list(ast.walk(probe_tree))
    ann_roots = [
        ann
        for n in all_nodes
        for ann in (
            getattr(n, "annotation", None),
            getattr(n, "returns", None),
        )
        if isinstance(ann, ast.AST)
    ]
    for n in all_nodes:
        if isinstance(n, ast.Call) and n.args:
            func = n.func
            func_name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else "")
            )
            if func_name == "cast":
                ann_roots.append(n.args[0])
    str_ann_text = "\n".join(
        sub.value
        for root in ann_roots
        for sub in ast.walk(root)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    )
    return {
        n.id for n in all_nodes if isinstance(n, ast.Name)
    } | set(re.findall(r"\b[A-Za-z_]\w*\b", f"{return_type}\n{str_ann_text}"))


_SYNTHESIS_DISALLOWED_HAZARDS: frozenset[str] = frozenset(
    {
        "type_param_closure_capture",
        "pathological_ast",
        "stale_source_code",
        "invalid_unit_columns",
        "deleted_output",
        "severed_closure_cell",
        "unavailable_source_for_closure_mode",
    }
)


def _has_blocking_synthesis_hazards(
    hazards1: AbstractSet[str],
    hazards2: AbstractSet[str],
    *,
    u2_allows_placeholder: bool,
    extra_hazards: Iterable[str] = (),
) -> bool:
    """Returns True when unit or pair hazards block helper synthesis."""
    if "unavailable_source_placeholder" in hazards1 or (
        "unavailable_source_placeholder" in hazards2 and not u2_allows_placeholder
    ):
        return True
    return any(
        h in _SYNTHESIS_DISALLOWED_HAZARDS
        for h in (*hazards1, *hazards2, *extra_hazards)
    )


class _PreparedSynthesisLines(NamedTuple):
    """Normalized units, token lines, pragmas, and common body lines for helper synthesis."""

    u1: Dict[str, Any]
    u2: Dict[str, Any]
    orig_lines1: List[str]
    orig_lines2: List[str]
    common_lines: List[str]
    pragmas1: List[Dict[str, Any]]
    pragmas2: List[Dict[str, Any]]
    helper_name: str
    is_expr_unit: bool
    tok2_is_placeholder: bool


class _ResolvedSynthesisScopes(NamedTuple):
    """Evaluated per-unit and pair scopes plus parsed ASTs for helper synthesis."""

    scope1: Dict[str, Any]
    scope2: Dict[str, Any]
    scope: Dict[str, Any]
    tree1: Optional[ast.AST]
    tree2: Optional[ast.AST]
    effective_strictness: str
    is_sub: bool


class _ResolvedSynthesisBinding(NamedTuple):
    """Resolved helper binding strategy, receiver metadata, and filtered inputs."""

    effective_binding: str
    is_static_clone: bool
    is_class_receiver: bool
    indent: str
    inputs: List[str]
    inputs2: List[str]


def _prepare_synthesis_lines(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    *,
    source_text1: Optional[str],
    source_text2: Optional[str],
    repo_root: Optional[str],
    helper_name: Optional[str],
    preserve_pragmas: bool,
) -> Optional[_PreparedSynthesisLines]:
    """Validates schemas, extracts unit token lines, and builds initial common lines."""
    validate_unit_schema(u1)
    validate_unit_schema(u2)
    norm_u1 = dict(u1)
    norm_u2 = dict(u2)
    for u_target, src_override in ((norm_u1, source_text1), (norm_u2, source_text2)):
        if src_override is not None and (
            not isinstance(u_target.get("source_text"), str)
            or not u_target["source_text"].strip()
        ):
            u_target["source_text"] = src_override
    if not _align_pair_expr_kinds(norm_u1, norm_u2):
        return None
    norm_u1 = resolve_unit_inferred_col_bounds(norm_u1, repo_root=repo_root).unit
    norm_u2 = resolve_unit_inferred_col_bounds(norm_u2, repo_root=repo_root).unit
    tok1 = _extract_unit_token_lines_with_state(norm_u1, repo_root=repo_root)
    tok2 = _extract_unit_token_lines_with_state(norm_u2, repo_root=repo_root)
    if tok1.lines is None or tok2.lines is None or tok1.is_placeholder:
        return None
    lines1 = tok1.lines
    lines2 = tok2.lines

    base_name1 = _base_unit_name(norm_u1)
    base_name2 = _base_unit_name(norm_u2)
    resolved_name = helper_name or (
        f"_shared_{base_name1}"
        if base_name1 == base_name2
        else f"_shared_{base_name1}_{base_name2}"
    )

    orig_lines1 = list(lines1)
    orig_lines2 = list(lines2)
    pragmas1: List[Dict[str, Any]] = []
    pragmas2: List[Dict[str, Any]] = []
    if preserve_pragmas:
        pragmas1 = extract_unit_comments_and_pragmas(
            "\n".join(orig_lines1), 1, len(orig_lines1)
        )
        pragmas2 = extract_unit_comments_and_pragmas(
            "\n".join(orig_lines2), 1, len(orig_lines2)
        )

    is_expr_unit = norm_u1.get("kind") in ("comprehension", "complex_expr")
    if is_expr_unit:
        if not (
            _is_valid_expr_unit_lines(norm_u1, orig_lines1)
            and _is_valid_expr_unit_lines(norm_u2, orig_lines2)
        ):
            return None
        lines1 = _strip_stmt_target_lines(lines1)
        lines2 = _strip_stmt_target_lines(lines2)

    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    common_lines: List[str] = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            common_lines.extend(lines1[i1:i2])
    if not common_lines or len(common_lines) < len(lines1):
        common_lines = lines1
    if preserve_pragmas:
        common_lines = _attach_pragmas(common_lines, pragmas1 + pragmas2)

    return _PreparedSynthesisLines(
        u1=norm_u1,
        u2=norm_u2,
        orig_lines1=orig_lines1,
        orig_lines2=orig_lines2,
        common_lines=common_lines,
        pragmas1=pragmas1,
        pragmas2=pragmas2,
        helper_name=resolved_name,
        is_expr_unit=is_expr_unit,
        tok2_is_placeholder=tok2.is_placeholder,
    )


def _resolve_synthesis_tree(
    given_tree: Optional[ast.AST],
    given_src: Optional[str],
    unit: Dict[str, Any],
    *,
    repo_root: Optional[str],
) -> Optional[ast.AST]:
    """Resolves or parses the module AST for a single synthesis unit."""
    if given_tree is not None:
        return given_tree
    raw_text = given_src or _load_unit_file_text(unit, repo_root=repo_root)
    if raw_text:
        return _get_or_parse_tree(raw_text)
    return None


def _resolve_synthesis_scopes(
    prep: _PreparedSynthesisLines,
    *,
    repo_root: Optional[str],
    source_text1: Optional[str],
    source_text2: Optional[str],
    tree1: Optional[ast.AST],
    tree2: Optional[ast.AST],
    closure_strictness: Optional[str],
    non_suppressing_ctx_managers: Optional[Sequence[str]],
    outputs: Optional[List[str]],
    outputs2: Optional[List[str]],
    scope1: Optional[Dict[str, Any]],
    scope2: Optional[Dict[str, Any]],
    scope: Optional[Dict[str, Any]],
    scopes_lhs_pruned: bool = False,
) -> Optional[_ResolvedSynthesisScopes]:
    """Computes and validates per-unit and pair variable scopes via _pair_eval."""
    u1, u2 = prep.u1, prep.u2
    resolved_tree1 = _resolve_synthesis_tree(
        tree1, source_text1, u1, repo_root=repo_root
    )
    resolved_tree2 = _resolve_synthesis_tree(
        tree2, source_text2, u2, repo_root=repo_root
    )
    for u_item, s_text, t_item in (
        (u1, source_text1, resolved_tree1),
        (u2, source_text2, resolved_tree2),
    ):
        _populate_unit_receiver_metadata(
            u_item, repo_root=repo_root, source_text=s_text, tree=t_item
        )

    pair_policy = resolve_pair_closure_policy(
        u1,
        u2,
        closure_strictness=closure_strictness,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
    )
    eff_non_sup1 = pair_policy.unit1.non_suppressing_ctx_managers
    eff_non_sup2 = pair_policy.unit2.non_suppressing_ctx_managers
    effective_strictness = pair_policy.effective_mode
    has_precomputed_scopes = (
        scope1 is not None
        and scope2 is not None
        and scope is not None
        and scope_matches_closure_mode(
            scope1,
            effective_strictness,
            non_suppressing_ctx_managers=eff_non_sup1,
        )
        and scope_matches_closure_mode(
            scope2,
            effective_strictness,
            non_suppressing_ctx_managers=eff_non_sup2,
        )
        and scope_matches_closure_mode(
            scope,
            effective_strictness,
            non_suppressing_ctx_managers=sorted(set(eff_non_sup1) & set(eff_non_sup2)),
        )
    )
    if has_precomputed_scopes and scope1 is not None and scope2 is not None:
        resolved_s1 = copy.deepcopy(scope1)
        resolved_s2 = copy.deepcopy(scope2)
    else:
        pair_eval = evaluate_clone_pair_scopes(
            u1,
            u2,
            scope1=scope1,
            scope2=scope2,
            closure_strictness=closure_strictness,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
            source_text1=source_text1,
            source_text2=source_text2,
            tree1=resolved_tree1,
            tree2=resolved_tree2,
            repo_root=repo_root,
            copy_scopes=True,
            isolate_telemetry=False,
        )
        resolved_s1, resolved_s2 = pair_eval.scope1, pair_eval.scope2
    for u_item, s_item in ((u1, resolved_s1), (u2, resolved_s2)):
        if u_item.get("enclosing_is_async") and s_item.get("has_yield"):
            s_item["is_async"] = True
            u_item["is_async"] = True

    is_sub1 = is_subroutine_unit(u1, source_text=source_text1, tree=resolved_tree1)
    is_sub2 = is_subroutine_unit(u2, source_text=source_text2, tree=resolved_tree2)
    if is_sub1 != is_sub2:
        return None
    hazards1 = set(resolved_s1.get("control_flow_hazards", ()))
    hazards2 = set(resolved_s2.get("control_flow_hazards", ()))
    u2_allows_placeholder = allows_u2_source_placeholder(
        is_subroutine=is_sub1,
        requires_u2_edit=False,
        source_text2=source_text2,
        u2=u2,
        repo_root=repo_root,
    )
    if (
        (prep.tok2_is_placeholder and not u2_allows_placeholder)
        or _has_blocking_synthesis_hazards(
            hazards1, hazards2, u2_allows_placeholder=u2_allows_placeholder
        )
    ):
        return None

    if (
        has_precomputed_scopes
        and scope is not None
        and (not prep.is_expr_unit or scopes_lhs_pruned)
    ):
        if prep.is_expr_unit and (outputs is None or outputs2 is None):
            u1.pop("outputs", None)
            u2.pop("outputs", None)
            u1.pop("precomputed_outputs", None)
            u2.pop("precomputed_outputs", None)
        resolved_pair = dict(scope)
    else:
        if not _prune_pair_lhs_and_augassign_scopes(
            u1, u2, resolved_s1, resolved_s2, prep.orig_lines1, prep.orig_lines2
        ):
            return None
        resolved_pair = analyze_unit_variable_scope(
            u1,
            u2,
            repo_root=repo_root,
            tree1=resolved_tree1,
            tree2=resolved_tree2,
            scope1=resolved_s1,
            scope2=resolved_s2,
            closure_strictness=effective_strictness,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        )
    if _has_blocking_synthesis_hazards(
        hazards1,
        hazards2,
        u2_allows_placeholder=u2_allows_placeholder,
        extra_hazards=resolved_pair.get("control_flow_hazards", ()),
    ):
        return None
    if resolved_s1.get("is_async") or resolved_s2.get("is_async"):
        resolved_pair["is_async"] = True

    return _ResolvedSynthesisScopes(
        scope1=resolved_s1,
        scope2=resolved_s2,
        scope=resolved_pair,
        tree1=resolved_tree1,
        tree2=resolved_tree2,
        effective_strictness=effective_strictness,
        is_sub=is_sub1,
    )


def _infer_unit_receiver_kind(
    unit: Dict[str, Any],
    unit_scope: Dict[str, Any],
    *,
    receiver_kind: Optional[str],
    is_static: bool,
) -> str:
    """Infers the effective receiver kind ('instance', 'class', 'static', or 'none') for a unit."""
    if unit.get("receiver_kind"):
        return str(unit["receiver_kind"])
    if "receiver_kind" in unit and unit.get("receiver_kind") is None:
        return "none"
    if receiver_kind:
        return receiver_kind
    if unit.get("kind") in ("closure", "class", "comprehension", "data_table", "complex_expr"):
        return "none"
    if unit.get("is_static"):
        return "static"
    binding_kind = unit_scope.get("binding_kind")
    if binding_kind in ("static", "class", "instance"):
        return str(binding_kind)
    if is_static:
        return "static"
    if unit.get("enclosing_class"):
        return "instance"
    return "none"


def _resolve_synthesis_binding(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scopes: _ResolvedSynthesisScopes,
    *,
    method_binding: str,
    indent: str,
    is_static: bool,
    receiver_kind: Optional[str],
    repo_root: Optional[str],
) -> Optional[_ResolvedSynthesisBinding]:
    """Validates pair semantic compatibility and resolves method/module binding + inputs."""
    scope1, scope2, scope = scopes.scope1, scopes.scope2, scopes.scope
    enc1, enc2 = u1.get("enclosing_class"), u2.get("enclosing_class")
    f1 = normalize_path_string(unit_file_str(u1))
    f2 = normalize_path_string(unit_file_str(u2))
    enc1_start = u1.get("enclosing_class_start")
    enc2_start = u2.get("enclosing_class_start")
    same_file = _is_same_file_path(f1, f2, repo_root=repo_root)
    is_same_class = bool(
        enc1
        and enc2
        and enc1 == enc2
        and same_file
        and (enc1_start is None or enc2_start is None or enc1_start == enc2_start)
    )

    k1 = _infer_unit_receiver_kind(
        u1, scope1, receiver_kind=receiver_kind, is_static=is_static
    )
    k2 = _infer_unit_receiver_kind(
        u2, scope2, receiver_kind=receiver_kind, is_static=is_static
    )
    receiver_kinds_differ = bool(k1 != k2)
    if receiver_kinds_differ and (
        _has_receiver_reference(u1, scope1, repo_root=repo_root)
        or _has_receiver_reference(u2, scope2, repo_root=repo_root)
    ):
        return None
    compat_check = check_pair_core_hazards(
        scope1,
        scope2,
        is_same_file=is_same_class or same_file,
        check_stage="compatibility",
        pair_scope=scope,
    )
    if compat_check.skip_reason is not None:
        return None

    rec1 = u1.get("receiver_param") or ("cls" if k1 == "class" else "self")
    rec2 = u2.get("receiver_param") or ("cls" if k2 == "class" else "self")
    if (
        _normalize_receiver_attrs(scope1.get("attrs_read", []), rec1)
        != _normalize_receiver_attrs(scope2.get("attrs_read", []), rec2)
        or _normalize_receiver_attrs(scope1.get("attrs_written", []), rec1)
        != _normalize_receiver_attrs(scope2.get("attrs_written", []), rec2)
    ):
        return None

    is_static_clone = bool(
        is_static or receiver_kind == "static" or (k1 == "static" and k2 == "static")
    )
    is_class_receiver = bool(
        receiver_kind == "class" or (k1 == "class" and k2 == "class")
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
    if (
        (has_super and (effective_binding == "module" or is_static_clone))
        or (has_mangled and (effective_binding != "method" or not is_same_class or is_static_clone))
        or (
            (
                scope1.get("has_class_scoped_type_params")
                or scope2.get("has_class_scoped_type_params")
            )
            and (effective_binding != "method" or not is_same_class)
        )
    ):
        return None

    inputs = list(scope["inputs"])
    inputs2 = list(scope2.get("inputs", []))
    if effective_binding == "module":
        inputs = _prune_unshared_receivers(
            inputs, u1, u2, scope1, scope2, repo_root=repo_root
        )
        inputs2 = _prune_unshared_receivers(
            inputs2, u2, u1, scope2, scope1, repo_root=repo_root
        )
    resolved_indent = "    " if (effective_binding == "method" and not indent) else indent
    return _ResolvedSynthesisBinding(
        effective_binding=effective_binding,
        is_static_clone=is_static_clone,
        is_class_receiver=is_class_receiver,
        indent=resolved_indent,
        inputs=inputs,
        inputs2=inputs2,
    )


def _resolve_synthesis_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scopes: _ResolvedSynthesisScopes,
    *,
    outputs: Optional[List[str]],
    outputs2: Optional[List[str]],
    source_text1: Optional[str],
    source_text2: Optional[str],
    repo_root: Optional[str],
    non_suppressing_ctx_managers: Optional[Sequence[str]],
) -> Optional[Tuple[List[str], List[str]]]:
    """Resolves helper return-output variable lists with global/nonlocal filtering."""
    scope1, scope2, scope = scopes.scope1, scopes.scope2, scopes.scope
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
            tree1=scopes.tree1,
            tree2=scopes.tree2,
            repo_root=repo_root,
            closure_strictness=scopes.effective_strictness,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
            require_pairing=False,
        )
        if resolved_outs is None:
            return None
        eff_outputs, u2_outs = resolved_outs

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
        return None
    return helper_outputs, helper_outputs2


def _normalize_synthesis_body_lines(
    common_lines: List[str],
    *,
    scope: Dict[str, Any],
    helper_outputs: List[str],
    inputs: Sequence[str],
    conditional_outs: AbstractSet[str],
    is_expr_unit: bool,
    is_async: bool,
    indent: str,
    step: Optional[str],
    preserve_pragmas: bool,
    all_pragmas: Sequence[Dict[str, Any]],
) -> Optional[Tuple[List[str], List[str]]]:
    """Dedents body lines, prepends global/conditional guards, and adds return statements."""
    lines = [
        ln.rstrip("\r\n")
        for ln in split_source_lines(dedent_source_text("\n".join(common_lines)))
    ]
    prefix_stmts: List[str] = []
    if scope.get("globals"):
        g_vars = sorted(set(scope["globals"]))
        if not any(ln.strip().startswith("global ") for ln in lines):
            prefix_stmts.append(f"global {', '.join(g_vars)}")
    for out_var in helper_outputs:
        if out_var in conditional_outs and out_var not in inputs:
            prefix_stmts.append(f"{out_var} = None")
    if prefix_stmts:
        lines = prefix_stmts + lines

    has_trailing_return = any(
        ln.strip() == "return"
        or (
            ln.strip().startswith("return")
            and len(ln.strip()) > 6
            and ln.strip()[6] in (" ", "\t", "(", "#")
        )
        for ln in lines[-3:]
    )
    final_outputs = list(helper_outputs)
    if is_expr_unit:
        if not has_trailing_return:
            full_code = "\n".join(lines).strip()
            stripped_expr: Optional[str] = None
            try:
                parsed_body = _parse_dedented_stmts("\n".join(lines))
                if len(parsed_body) > 1:
                    return None
                if len(parsed_body) == 1:
                    stmt_node = parsed_body[0]
                    if isinstance(stmt_node, (ast.Assign, ast.AugAssign)):
                        stripped_expr = ast.unparse(stmt_node.value)
                    elif isinstance(stmt_node, ast.AnnAssign) and stmt_node.value is not None:
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
                        return None
            except (SyntaxError, IndexError, ValueError, RecursionError):
                pass

            if stripped_expr is not None:
                stripped_lines = [
                    ln.rstrip("\r\n") for ln in split_source_lines(stripped_expr)
                ]
                if len(stripped_lines) == 1:
                    lines = [f"return {stripped_lines[0]}"]
                else:
                    eff_step = step or _detect_indent_step(indent)
                    lines = ["return ("] + _indent_code_lines(stripped_lines, eff_step) + [")"]
            elif len(lines) == 1:
                first_expr = lines[0].strip()
                m_assign = _ASSIGN_STMT_RE.match(first_expr)
                if m_assign:
                    first_expr = m_assign.group(1).strip()
                lines = [f"return {first_expr}"]
            else:
                eff_step = step or _detect_indent_step(indent)
                lines = ["return ("] + _indent_code_lines(lines, eff_step) + [")"]
            if preserve_pragmas:
                lines = _attach_pragmas(lines, all_pragmas)
        final_outputs = []
    elif (
        not has_trailing_return
        and final_outputs
        and not (bool(scope.get("has_yield")) and is_async)
    ):
        ret_expr = ", ".join(final_outputs) if len(final_outputs) >= 2 else final_outputs[0]
        lines = lines + [f"return {ret_expr}"]
    return lines, final_outputs


def _format_active_type_params(
    scopes: _ResolvedSynthesisScopes,
    *,
    func_keyword: str,
    params_str: str,
    return_type: str,
    indented_body: str,
) -> str:
    """Formats the PEP 695 type-parameter bracket clause retained for the synthesized helper."""
    if scopes.is_sub:
        return ""
    tp_list = list(
        scopes.scope.get("type_params")
        or scopes.scope1.get("type_params")
        or scopes.scope2.get("type_params")
        or []
    )
    if not tp_list:
        return ""
    tp1_idents = {
        tp.split(":", 1)[0].split("=", 1)[0].lstrip("*").strip()
        for tp in (scopes.scope1.get("type_params") or [])
    }
    tp2_idents = {
        tp.split(":", 1)[0].split("=", 1)[0].lstrip("*").strip()
        for tp in (scopes.scope2.get("type_params") or [])
    }
    common_tp_idents = (
        (tp1_idents & tp2_idents)
        if (tp1_idents and tp2_idents)
        else (tp1_idents | tp2_idents)
    )
    probe_src = (
        f"{func_keyword} _probe({params_str}) -> {return_type}:\n"
        f"{indented_body}\n"
    )
    active_idents = common_tp_idents | _extract_probe_active_idents(
        probe_src, return_type
    )
    kept_tps: List[str] = []
    while True:
        next_tps = [
            tp
            for tp in tp_list
            if tp.split(":", 1)[0].split("=", 1)[0].lstrip("*").strip() in active_idents
        ]
        if len(next_tps) == len(kept_tps):
            tp_list = next_tps
            break
        kept_tps = next_tps
        for tp in kept_tps:
            head_len = len(tp.split(":", 1)[0].split("=", 1)[0])
            active_idents.update(re.findall(r"\b[A-Za-z_]\w*\b", tp[head_len:]))
    return f"[{', '.join(tp_list)}]" if tp_list else ""


def _compile_source_safely(source: str, filename: str) -> bool:
    """Compiles synthesized Python source, returning False on syntax/parser-stack failures."""
    try:
        compile(source, filename, "exec")
    except (SyntaxError, ValueError, RecursionError):
        return False
    except MemoryError as exc:
        if not is_cpython_parser_stack_overflow(exc, source=source):
            raise
        return False
    return True


def _finalize_helper_source(
    helper_def: str,
    *,
    scope: Dict[str, Any],
    indent: str,
    include_imports: bool,
) -> str:
    """Validates compiled syntax and optionally prepends typing/local import statements."""
    hazards = scope.get("control_flow_hazards", [])
    has_naked_ctrl = any(h in ("naked_break", "naked_continue") for h in hazards)
    if not has_naked_ctrl:
        probe_src = f"if 1:\n{helper_def}" if indent else helper_def
        if not _compile_source_safely(probe_src, "<synthesized_helper>"):
            return ""
    if not include_imports:
        return helper_def

    needed_typing = _extract_required_typing_imports(helper_def)
    import_header_parts: List[str] = []
    if needed_typing:
        import_header_parts.append(f"from typing import {', '.join(sorted(needed_typing))}")
    for loc_imp in scope.get("local_imports", []):
        if loc_imp not in import_header_parts:
            import_header_parts.append(loc_imp)
    if not import_header_parts:
        return helper_def
    imports_text = "\n".join(import_header_parts)
    if not _compile_source_safely(imports_text, "<synthesized_imports>"):
        return ""
    return imports_text + "\n\n\n" + helper_def


@_with_analysis_session
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
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    outputs: Optional[List[str]] = None,
    outputs2: Optional[List[str]] = None,
    scope1: Optional[Dict[str, Any]] = None,
    scope2: Optional[Dict[str, Any]] = None,
    scope: Optional[Dict[str, Any]] = None,
    scopes_lhs_pruned: bool = False,
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
        closure_strictness: Closure isolation strictness ("strict", "pedantic",
            or "lenient"). Per-unit ``unit["closure_strictness"]`` fields are validated
            strictly via :func:`validate_unit_schema` (raising ``ValueError`` if invalid),
            whereas an unrecognized string passed to this keyword parameter logs a
            deduplicated warning and falls back via :func:`resolve_closure_strictness_mode`.
        outputs: Optional pre-resolved output variable names for u1.
        outputs2: Optional pre-resolved output variable names for u2.
    """
    prep = _prepare_synthesis_lines(
        u1,
        u2,
        source_text1=source_text1,
        source_text2=source_text2,
        repo_root=repo_root,
        helper_name=helper_name,
        preserve_pragmas=preserve_pragmas,
    )
    if prep is None:
        return ""
    scopes = _resolve_synthesis_scopes(
        prep,
        repo_root=repo_root,
        source_text1=source_text1,
        source_text2=source_text2,
        tree1=tree1,
        tree2=tree2,
        closure_strictness=closure_strictness,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        outputs=outputs,
        outputs2=outputs2,
        scope1=scope1,
        scope2=scope2,
        scope=scope,
        scopes_lhs_pruned=scopes_lhs_pruned,
    )
    if scopes is None:
        return ""
    binding = _resolve_synthesis_binding(
        prep.u1,
        prep.u2,
        scopes,
        method_binding=method_binding,
        indent=indent,
        is_static=is_static,
        receiver_kind=receiver_kind,
        repo_root=repo_root,
    )
    if binding is None:
        return ""
    resolved_outs = _resolve_synthesis_outputs(
        prep.u1,
        prep.u2,
        scopes,
        outputs=outputs,
        outputs2=outputs2,
        source_text1=source_text1,
        source_text2=source_text2,
        repo_root=repo_root,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
    )
    if resolved_outs is None:
        return ""
    helper_outputs, helper_outputs2 = resolved_outs

    meta1 = {p["name"].lstrip("*"): p for p in scopes.scope1.get("param_details", [])}
    meta2 = {p["name"].lstrip("*"): p for p in scopes.scope2.get("param_details", [])}
    rec_param = prep.u1.get("receiver_param") or prep.u2.get("receiver_param")
    params = _format_helper_parameters(
        binding.inputs,
        meta1,
        meta2,
        effective_binding=binding.effective_binding,
        is_static_clone=binding.is_static_clone,
        is_class_receiver=binding.is_class_receiver,
        type_merge_strategy=type_merge_strategy,
        inputs2=binding.inputs2 if len(binding.inputs2) == len(binding.inputs) else None,
        receiver_param=rec_param,
        positional_target_map=prep.is_expr_unit,
    )
    resolved_ret = _merge_types(
        scopes.scope1.get("return_type"),
        scopes.scope2.get("return_type"),
        type_merge_strategy,
    )
    conditional_outs = set(scopes.scope.get("conditional_outputs", []))
    is_async = bool(scopes.scope.get("is_async", False))
    func_keyword = "async def" if is_async else "def"
    await_prefix = "await " if is_async else ""
    return_type = _infer_helper_return_type(
        resolved_ret,
        helper_outputs,
        conditional_outs,
        scopes.scope,
        meta1,
        meta2,
        type_merge_strategy=type_merge_strategy,
        is_async=is_async,
        is_subroutine=scopes.is_sub,
        unit_kind=prep.u1.get("kind"),
        outputs2=(
            helper_outputs2 if len(helper_outputs2) == len(helper_outputs) else None
        ),
    )
    params_str = ", ".join(params) if params else "*args: Any, **kwargs: Any"

    body_result = _normalize_synthesis_body_lines(
        prep.common_lines,
        scope=scopes.scope,
        helper_outputs=helper_outputs,
        inputs=binding.inputs,
        conditional_outs=conditional_outs,
        is_expr_unit=prep.is_expr_unit,
        is_async=is_async,
        indent=binding.indent,
        step=step,
        preserve_pragmas=preserve_pragmas,
        all_pragmas=prep.pragmas1 + prep.pragmas2,
    )
    if body_result is None:
        return ""
    common_lines, helper_outputs = body_result

    eff_step = _detect_indent_step(binding.indent) if step is None else step
    body_indent = binding.indent + eff_step
    sub_indent = body_indent + eff_step
    if binding.effective_binding == "method":
        rec_call = prep.u1.get("receiver_param") or prep.u2.get("receiver_param")
        if binding.is_class_receiver:
            call_target = f"{rec_call or 'cls'}.{prep.helper_name}"
            dec_prefix = f"{binding.indent}@classmethod\n"
        elif binding.is_static_clone:
            call_target = f"__class__.{prep.helper_name}"
            dec_prefix = f"{binding.indent}@staticmethod\n"
        else:
            call_target = f"{rec_call or 'self'}.{prep.helper_name}"
            dec_prefix = ""
    else:
        call_target = prep.helper_name
        dec_prefix = ""

    docstring_str = _format_helper_docstring(
        call_target,
        helper_outputs,
        scopes.scope,
        is_async=is_async,
        doc_indent=body_indent,
        sub_indent=sub_indent,
        await_prefix=await_prefix,
    )
    indented_body = "\n".join(_indent_code_lines(common_lines, body_indent))
    type_params_str = _format_active_type_params(
        scopes,
        func_keyword=func_keyword,
        params_str=params_str,
        return_type=return_type,
        indented_body=indented_body,
    )
    helper_def = (
        f"{dec_prefix}"
        f"{binding.indent}{func_keyword} {prep.helper_name}{type_params_str}"
        f"({params_str}) -> {return_type}:\n"
        f"{docstring_str}\n"
        f"{indented_body}\n"
    )
    return _finalize_helper_source(
        helper_def,
        scope=scopes.scope,
        indent=binding.indent,
        include_imports=include_imports,
    )
