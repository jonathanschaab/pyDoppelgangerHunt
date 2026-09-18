"""Shared helper function synthesis, parameter ordering, docstrings, and typing imports."""

from __future__ import annotations

import ast
import difflib
import re
import textwrap
import typing
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydoppelgangerhunt.config import normalize_path_string
from pydoppelgangerhunt.reporters import extract_unit_source_code
from pydoppelgangerhunt.fixer.binding import (
    _base_unit_name,
    _has_receiver_reference,
    _is_same_file_path,
    _populate_unit_receiver_metadata,
    _prune_unshared_receivers,
    _resolve_effective_binding,
)
from pydoppelgangerhunt.fixer.scope import (
    _normalize_receiver_attrs,
    _rank_param_kind,
    dispatch_analyze_unit_variable_scope as analyze_unit_variable_scope,
)
from pydoppelgangerhunt.fixer.source import (
    _detect_indent_step,
    _extract_unit_body_lines,
    _slice_unit_token_lines,
    extract_unit_comments_and_pragmas,
)

def _format_call_arguments(
    inputs: List[str],
    param_details: List[Dict[str, Any]],
    receiver_to_omit: Optional[str] = None,
    target_inputs: Optional[List[str]] = None,
) -> str:
    """Formats argument strings for helper call sites, preserving keyword-only, vararg, and kwarg syntax."""
    param_map: Dict[str, str] = {
        p["name"].lstrip("*"): p.get("kind", "pos") for p in param_details
    }
    sorted_inputs = sorted(inputs, key=lambda v: _rank_param_kind(v, param_map))
    if target_inputs is not None and len(target_inputs) == len(inputs):
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
            or (receiver_to_omit == "receivers" and clean_t in ("self", "cls"))
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

def _merge_types(t1: Optional[str], t2: Optional[str], strategy: str) -> str:
    """Merges two type annotations according to the specified merge strategy."""
    if t1 and t2:
        if t1 == t2:
            return t1
        if strategy == "union":
            return f"Union[{t1}, {t2}]"
        return "Any"
    return t1 or t2 or "Any"

TYPING_SYMBOLS: Set[str] = set(typing.__all__)


def _extract_required_typing_imports(signature_or_func_text: str) -> List[str]:
    """Identifies typing symbols referenced in type annotations via AST inspection."""
    text = signature_or_func_text.strip()
    tree: Optional[ast.AST] = None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        pass

    if tree is None and "->" in text:
        parts = text.split("->", 1)
        params_part = parts[0].strip()
        if params_part.startswith("(") and params_part.endswith(")"):
            params_part = params_part[1:-1].strip()
        ret_part = parts[1].strip()
        try:
            tree = ast.parse(f"def _sig_wrapper({params_part}) -> {ret_part}: pass")
        except SyntaxError:
            pass

    if tree is None:
        try:
            tree = ast.parse(f"def _sig_wrapper({text}): pass")
        except SyntaxError:
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
) -> List[str]:
    """Formats helper function parameters with type annotations and default values."""
    params: List[str] = []
    seen_kwonly = False

    # Extract parameter descriptors
    descriptors: List[Dict[str, Any]] = []
    for idx, var in enumerate(inputs):
        m1 = meta1.get(var, {})
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
    all_receivers = {primary_receiver, "self", "cls"}

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
        resolved_type = desc["type"]
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


def _infer_helper_return_type(
    resolved_ret: str,
    helper_outputs: List[str],
    conditional_outs: Set[str],
    scope: Dict[str, Any],
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    type_merge_strategy: str = "fallback_any",
    is_async: bool = False,
    unit_kind: Optional[str] = None,
    outputs2: Optional[List[str]] = None,
) -> str:
    """Infers the return type annotation for a synthesized shared helper function."""
    if scope.get("has_yield"):
        if resolved_ret != "Any":
            return resolved_ret
        inferred_yield_type = None
        for kind, name in scope.get("yield_expr_names", []):
            if name.startswith(":literal:"):
                inferred_yield_type = name[len(":literal:") :]
                break
            m_t = meta1.get(name, {}).get("type") or meta2.get(name, {}).get("type")
            if not m_t:
                continue
            if kind == "yield":
                inferred_yield_type = m_t
                break
            if kind == "yield_from":
                for prefix in (
                    "Iterator[", "Iterable[", "List[", "Sequence[", "Set[", "Tuple[", "Collection[",
                    "list[", "set[", "tuple[", "sequence[", "iterable[", "iterator[",
                ):
                    if m_t.startswith(prefix) and m_t.endswith("]"):
                        inner = m_t[len(prefix) : -1].strip()
                        if "," in inner:
                            inner = inner.split(",")[0].strip()
                        inferred_yield_type = inner
                        break
                if inferred_yield_type:
                    break
        iter_name = "AsyncIterator" if is_async else "Iterator"
        if inferred_yield_type:
            return f"{iter_name}[{inferred_yield_type}]"
        return f"{iter_name}[Any]"

    if len(helper_outputs) >= 2:
        out_types: List[str] = []
        for idx, out_var in enumerate(helper_outputs):
            t1 = meta1.get(out_var, {}).get("type")
            m2 = meta2.get(out_var, {})
            if not m2 and outputs2 and idx < len(outputs2):
                m2 = meta2.get(outputs2[idx], {})
            t2 = m2.get("type")
            t_merged = _merge_types(t1, t2, type_merge_strategy)
            if out_var in conditional_outs:
                if not t_merged.startswith("Optional[") and "None" not in t_merged:
                    t_merged = f"Optional[{t_merged}]"
            out_types.append(t_merged)
        return f"Tuple[{', '.join(out_types)}]"

    if len(helper_outputs) == 1:
        out_var = helper_outputs[0]
        t1 = meta1.get(out_var, {}).get("type")
        m2 = meta2.get(out_var, {})
        if not m2 and outputs2 and len(outputs2) >= 1:
            m2 = meta2.get(outputs2[0], {})
        t2 = m2.get("type")
        out_t = _merge_types(t1, t2, type_merge_strategy)
        if out_var in conditional_outs:
            if not out_t.startswith("Optional[") and "None" not in out_t:
                out_t = f"Optional[{out_t}]"
        return_type = resolved_ret if resolved_ret != "Any" else out_t
        if out_var in conditional_outs:
            if not return_type.startswith("Optional[") and "None" not in return_type and return_type != "None":
                return_type = f"Optional[{return_type}]"
        return return_type

    if not helper_outputs and not scope.get("has_return") and resolved_ret == "Any":
        return "Any" if unit_kind in ("comprehension", "complex_expr") else "None"

    return resolved_ret


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
    """
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

    # Find common lines using difflib matching
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    common_lines: List[str] = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            common_lines.extend(lines1[i1:i2])

    if not common_lines or len(common_lines) < len(lines1):
        common_lines = lines1

    if preserve_pragmas:
        raw1 = "\n".join(lines1)
        raw2 = "\n".join(lines2)
        pragmas1 = extract_unit_comments_and_pragmas(raw1, 1, len(lines1))
        pragmas2 = extract_unit_comments_and_pragmas(raw2, 1, len(lines2))
        for p in pragmas1 + pragmas2:
            if p["is_pragma"]:
                p_text = p["text"].strip()
                if not any(p_text in ln for ln in common_lines):
                    for idx, ln in enumerate(common_lines):
                        if ln.strip() and not ln.strip().startswith("#"):
                            common_lines[idx] = ln.rstrip() + f"  {p_text}"
                            break

    _populate_unit_receiver_metadata(u1, repo_root=repo_root)
    _populate_unit_receiver_metadata(u2, repo_root=repo_root)

    # Variable scope analysis for concrete parameter signatures
    scope1 = analyze_unit_variable_scope(u1, repo_root=repo_root)
    scope2 = analyze_unit_variable_scope(u2, repo_root=repo_root)
    scope = analyze_unit_variable_scope(u1, u2, repo_root=repo_root)

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
        if u.get("kind") in ("closure", "class", "comprehension", "data_table", "complex_expr"):
            return "none"
        if u.get("is_static"):
            return "static"
        bk = u_scope.get("binding_kind")
        if bk in ("static", "class", "instance"):
            return str(bk)
        if receiver_kind:
            return receiver_kind
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

    if bool(scope1.get("is_async")) != bool(scope2.get("is_async")):
        return ""
    if bool(scope1.get("has_yield")) != bool(scope2.get("has_yield")):
        return ""
    if scope1.get("nonlocals") or scope2.get("nonlocals") or scope.get("nonlocals"):
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

    is_in_method = bool(
        u1.get("kind") not in ("comprehension", "complex_expr", "closure", "class")
        and u2.get("kind") not in ("comprehension", "complex_expr", "closure", "class")
        and k1 != "none"
        and k2 != "none"
    )

    effective_binding = _resolve_effective_binding(
        method_binding,
        is_same_class,
        is_static=is_static_clone,
        receiver_kinds_differ=receiver_kinds_differ,
        is_in_method=is_in_method,
        receiver_kind=k1,
    )

    has_super = bool(scope1.get("has_super") or scope2.get("has_super"))
    if has_super and (effective_binding == "module" or is_static_clone):
        return ""

    has_mangled = bool(scope1.get("has_mangled_names") or scope2.get("has_mangled_names"))
    if has_mangled and (effective_binding != "method" or not is_same_class or is_static_clone):
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
    )

    r1 = scope1.get("return_type")
    r2 = scope2.get("return_type")
    resolved_ret = _merge_types(r1, r2, type_merge_strategy)

    outputs = list(scope.get("outputs", []))
    conditional_outs = set(scope.get("conditional_outputs", []))
    is_async = scope.get("is_async", False)
    func_keyword = "async def" if is_async else "def"
    await_prefix = "await " if is_async else ""

    # Variables altered via global or nonlocal statements must not be captured in the helper's return tuple
    helper_outputs = [
        v for v in outputs
        if v not in scope.get("globals", [])
        and v not in scope.get("nonlocals", [])
    ]

    u2_outs = [
        v for v in scope2.get("outputs", [])
        if v not in scope2.get("globals", [])
        and v not in scope2.get("nonlocals", [])
    ]

    return_type = _infer_helper_return_type(
        resolved_ret,
        helper_outputs,
        conditional_outs,
        scope,
        meta1,
        meta2,
        type_merge_strategy=type_merge_strategy,
        is_async=is_async,
        unit_kind=u1.get("kind"),
        outputs2=u2_outs if len(u2_outs) == len(helper_outputs) else None,
    )

    params_str = ", ".join(params) if params else "*args: Any, **kwargs: Any"

    # Dedent common_lines first so relative block indentation is normalized
    common_lines = textwrap.dedent("\n".join(common_lines)).splitlines()

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
        if len(common_lines) == 1:
            common_lines = [f"return {common_lines[0].strip()}"]
        else:
            eff_step = step or _detect_indent_step(indent)
            expr_inner = [f"{eff_step}{ln}" for ln in common_lines]
            common_lines = ["return ("] + expr_inner + [")"]
        helper_outputs = []
    elif not has_trailing_return and helper_outputs and not scope.get("has_yield"):
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

    indented_body = "\n".join(f"{body_indent}{ln}" if ln.strip() else "" for ln in common_lines)
    if effective_binding == "method":
        if is_class_receiver:
            dec_prefix = f"{indent}@classmethod\n"
        elif is_static_clone:
            dec_prefix = f"{indent}@staticmethod\n"
        else:
            dec_prefix = ""
    else:
        dec_prefix = ""
    helper_def = (
        f"{dec_prefix}"
        f"{indent}{func_keyword} {helper_name}({params_str}) -> {return_type}:\n"
        f"{docstring_str}\n"
        f"{indented_body}\n"
    )

    if include_imports:
        needed_typing = _extract_required_typing_imports(helper_def)
        import_header_parts: List[str] = []
        if needed_typing:
            import_header_parts.append(f"from typing import {', '.join(sorted(needed_typing))}")
        for loc_imp in scope.get("local_imports", []):
            if loc_imp not in import_header_parts:
                import_header_parts.append(loc_imp)
        if import_header_parts:
            return "\n".join(import_header_parts) + "\n\n\n" + helper_def

    return helper_def
