"""Type annotation merging, typing import extraction, parameter formatting, and helper docstrings."""

from __future__ import annotations

import ast
import re
import typing
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydoppelgangerhunt._ast_compat import (
    is_type_alias_node,
    safe_parse_ast,
)
from pydoppelgangerhunt.fixer.source import is_subroutine_unit

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
    tree: Optional[ast.AST] = safe_parse_ast(text)

    if tree is None and "->" in text:
        parts = text.split("->", 1)
        params_part = parts[0].strip()
        if params_part.startswith("(") and params_part.endswith(")"):
            params_part = params_part[1:-1].strip()
        ret_part = parts[1].strip()
        tree = safe_parse_ast(
            f"def _sig_wrapper({params_part}) -> {ret_part}: pass"
        )

    if tree is None:
        tree = safe_parse_ast(f"def _sig_wrapper({text}): pass")

    if tree is None:
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
        return sorted(s for s in TYPING_SYMBOLS if s in tokens)

    has_defs = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign))
        or is_type_alias_node(n)
        for n in ast.walk(tree)
    )
    needed: Set[str] = set()
    tp_names: Set[str] = set()
    for node in ast.walk(tree):
        annotations: List[ast.AST] = []
        for tp in getattr(node, "type_params", ()):
            tp_name = getattr(tp, "name", None)
            if isinstance(tp_name, str):
                tp_names.add(tp_name)
            for attr in ("bound", "default_value"):
                tp_expr = getattr(tp, attr, None)
                if isinstance(tp_expr, ast.AST):
                    annotations.append(tp_expr)
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
        elif is_type_alias_node(node):
            alias_val = getattr(node, "value", None)
            if isinstance(alias_val, ast.AST):
                annotations.append(alias_val)
        elif not has_defs and isinstance(node, ast.Expr):
            annotations.append(node.value)

        for ann in annotations:
            for sub in ast.walk(ann):
                if isinstance(sub, ast.Name) and sub.id in TYPING_SYMBOLS:
                    needed.add(sub.id)
    return sorted(needed - tp_names)


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
            m2 = meta2.get(inputs2[idx], {})
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
            for kind, target in scope.get("yield_expr_names", []):
                if kind == "literal":
                    inferred_yield_type = _normalize_pipe_unions(target)
                    break
                m_t = meta1.get(target, {}).get("type") or meta2.get(target, {}).get("type")
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
