"""Control flow, enclosing block hazards, and downstream read collection.

Provides hazard checks for try/except/finally, with-blocks, and loop-carried
dependencies, as well as downstream read identification within lexical scopes.
"""

from __future__ import annotations

import ast
import builtins
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydoppelgangerhunt.fixer._df_cache import (
    _MAX_DOWNSTREAM_CACHE_SIZE,
    _build_downstream_cache_key,
    _coerce_string_frozenset,
    _downstream_cache_lock,
    _downstream_reads_cache,
    _extract_first_unit_coord,
    _extract_unit_end_col,
    _get_or_compute_scope_cached,
    _get_or_parse_tree,
    _get_valid_unit_bounds,
    _parse_source_tree,
    _resolve_unit_ast_end_col,
    _scope_loops_cache,
    _scope_try_with_cache,
    _store_lru_cache_entry,
    _warned_lenient_closure,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _collect_pre_unit_closures,
    _extract_scope_local_names,
    _extract_single_node_local_names,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _DownstreamReadVisitor,
    _StrictPostLineVisitor,
    _VisitorPassMode,
    _extract_scope_stmts,
    _extract_unit_live_in_names,
)
from pydoppelgangerhunt.fixer.source import _find_innermost_enclosing_node

logger = logging.getLogger(__name__)

_BUILTIN_NAMES: Set[str] = set(dir(builtins))


def _resolve_downstream_scope_node(
    source_text: str,
    unit: Dict[str, Any],
    tree: Optional[ast.AST],
) -> Optional[ast.AST]:
    """Resolves the innermost enclosing function AST node or falls back to module scope."""
    enc = _find_innermost_enclosing_node(
        source_text, unit, (ast.FunctionDef, ast.AsyncFunctionDef), tree=tree
    )
    if enc is not None:
        return enc[0]
    return _parse_source_tree(source_text, tree=tree)


def _find_enclosing_loops(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Detects loops enclosing the unit boundaries for loop-carried dependence analysis."""
    def _compute() -> List[Tuple[int, int, int]]:
        loops: List[Tuple[int, int, int]] = []
        for node in ast.walk(scope_node):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                l_start = getattr(node, "lineno", None) or 0
                l_end = getattr(node, "end_lineno", None) or l_start
                l_col = getattr(node, "col_offset", 0) or 0
                if l_start <= 0 or l_end <= 0:
                    continue
                loops.append((int(l_start), int(l_end), int(l_col)))
        return loops

    cached_loops = _get_or_compute_scope_cached(
        _scope_loops_cache, scope_node, _compute
    )
    return [
        (l_start, l_end)
        for l_start, l_end, l_col in cached_loops
        if l_start <= u_start
        and u_end <= l_end
        and (
            (l_start, l_end) != (u_start, u_end)
            or (u_start_col is not None and l_col < u_start_col)
        )
    ]


def _stmt_definitely_terminates(stmt: ast.stmt) -> bool:
    """Checks whether an AST statement unconditionally exits execution."""
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    if isinstance(stmt, ast.If):
        return (
            bool(stmt.orelse)
            and _block_definitely_terminates(stmt.body)
            and _block_definitely_terminates(stmt.orelse)
        )
    if isinstance(stmt, ast.Try):
        if stmt.finalbody and _block_definitely_terminates(stmt.finalbody):
            return True
        if stmt.handlers and _block_definitely_terminates(stmt.body):
            return all(_block_definitely_terminates(h.body) for h in stmt.handlers)
    if type(stmt).__name__ == "TryStar":
        finalbody = getattr(stmt, "finalbody", None)
        if finalbody and _block_definitely_terminates(finalbody):
            return True
        handlers = getattr(stmt, "handlers", [])
        body = getattr(stmt, "body", [])
        if handlers and _block_definitely_terminates(body):
            return all(_block_definitely_terminates(h.body) for h in handlers)
    return False


def _block_definitely_terminates(stmts: Sequence[ast.stmt]) -> bool:
    """Checks whether any statement in a sequence unconditionally terminates execution."""
    return any(_stmt_definitely_terminates(s) for s in stmts)


def _stmts_enclose_unit(stmts: Sequence[ast.stmt], u_start: int, u_end: int) -> bool:
    """Checks whether a sequence of AST statements spans and encloses the unit's line range."""
    if not stmts:
        return False
    valid_starts = [
        int(getattr(s, "lineno", 0))
        for s in stmts
        if getattr(s, "lineno", None) is not None and getattr(s, "lineno", 0) > 0
    ]
    if not valid_starts:
        return False
    s_start = min(valid_starts)
    valid_ends = [
        int(getattr(s, "end_lineno", None) or getattr(s, "lineno", None) or s_start)
        for s in stmts
        if getattr(s, "end_lineno", None) is not None or getattr(s, "lineno", None) is not None
    ]
    if not valid_ends:
        return False
    s_end = max(valid_ends)
    return s_start <= u_start and u_end <= s_end


def _nodes_load_any_name(nodes: Iterable[ast.AST], names: Set[str]) -> bool:
    """Checks whether any AST node in the sequence loads one of the given variable names."""
    for node in nodes:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Name)
                and isinstance(sub.ctx, ast.Load)
                and sub.id in names
            ):
                return True
    return False


def _get_scope_try_and_with_blocks(
    scope_node: ast.AST,
) -> List[ast.AST]:
    """Caches all Try, TryStar, With, and AsyncWith blocks within scope."""
    def _compute() -> List[ast.AST]:
        blocks: List[ast.AST] = []
        for node in ast.walk(scope_node):
            if node is scope_node:
                continue
            if (
                isinstance(node, (ast.Try, ast.With, ast.AsyncWith))
                or type(node).__name__ == "TryStar"
            ):
                blocks.append(node)
        return blocks

    return _get_or_compute_scope_cached(_scope_try_with_cache, scope_node, _compute)


def _execute_read_pass(
    scope_node: ast.AST,
    visitor_cls: type,
    u_start: int,
    u_end: int,
    u_end_col: Optional[int],
    candidates: Optional[Set[str]],
    enclosing_loops: Sequence[Tuple[int, int]],
    pass_mode: _VisitorPassMode,
    is_module: bool,
    u_start_col: Optional[int] = None,
) -> Set[str]:
    """Runs a downstream read visitor pass over a scope node and returns loaded names."""
    visitor = visitor_cls(
        u_start,
        u_end,
        u_end_col,
        candidates=candidates,
        enclosing_loops=enclosing_loops,
        pass_mode=pass_mode,
        is_module_scope=is_module,
        u_start_col=u_start_col,
    )
    visitor.outer_bound.update(
        _extract_single_node_local_names(scope_node, include_self_name=False)
    )
    for stmt in _extract_scope_stmts(scope_node):
        visitor.visit(stmt)
    return set(visitor.loaded)


def _scope_reads_outputs_after_line(
    scope_node: ast.AST,
    line: int,
    outputs: Set[str],
    end_col: Optional[int] = None,
    start_line: Optional[int] = None,
) -> bool:
    """Checks whether any statement in the scope loads outputs after the given line,
    including loop-carried reads if the statement sits inside an enclosing loop."""
    stmt_start = start_line if start_line is not None else line
    stmt_end = line
    enclosing_loops = _find_enclosing_loops(scope_node, stmt_start, stmt_end)
    is_module = isinstance(scope_node, ast.Module)

    after_reads = _execute_read_pass(
        scope_node,
        _StrictPostLineVisitor,
        stmt_end,
        stmt_end,
        end_col,
        outputs,
        enclosing_loops,
        _VisitorPassMode.AFTER_UNIT,
        is_module,
    )
    if not outputs.isdisjoint(after_reads):
        return True

    if enclosing_loops:
        loop_visitor = _DownstreamReadVisitor(
            stmt_start,
            stmt_end,
            end_col,
            candidates=outputs,
            enclosing_loops=enclosing_loops,
            pass_mode=_VisitorPassMode.LOOP_CARRIED,
            is_module_scope=is_module,
        )
        loop_visitor.outer_bound.update(
            _extract_single_node_local_names(scope_node, include_self_name=False)
        )
        for stmt in _extract_scope_stmts(scope_node):
            loop_visitor.visit(stmt)
        if not outputs.isdisjoint(loop_visitor.loaded):
            return True
        if loop_visitor.has_dynamic_read:
            return True
        live_ins = _extract_unit_live_in_names(
            scope_node,
            stmt_start,
            stmt_end,
            end_col,
            candidates=outputs,
        )
        loop_killed = (
            loop_visitor.unit_entry_killed
            if loop_visitor.unit_entry_killed is not None
            else set()
        )
        if not outputs.isdisjoint(live_ins - loop_killed):
            return True

    return False


def _compound_stmt_reads_outputs_after(
    scope_node: ast.AST,
    node: ast.AST,
    stmts: Sequence[ast.AST],
    outputs: Set[str],
) -> bool:
    """Checks whether outputs are read downstream of an enclosing compound statement."""
    end_raw = getattr(node, "end_lineno", None)
    end_col_raw = getattr(node, "end_col_offset", None)
    b_end = max(
        (
            int(getattr(s, "end_lineno", None) or getattr(s, "lineno", 0) or 0)
            for s in stmts
        ),
        default=0,
    )
    s_start = getattr(node, "lineno", None) or 1
    s_end = end_raw if isinstance(end_raw, int) else b_end
    s_end_col = end_col_raw if isinstance(end_col_raw, int) else None
    return _scope_reads_outputs_after_line(
        scope_node, s_end, outputs, end_col=s_end_col, start_line=s_start
    )


def _enclosing_try_reads_outputs(
    scope_node: Optional[ast.AST],
    u_start: int,
    u_end: int,
    outputs: Set[str],
) -> bool:
    """Returns True if unit is inside an enclosing try, else, or except block whose
    handlers/finally read outputs, or whose handlers fall through and outputs are read
    downstream of the try block, or inside any enclosing with or async with block whose
    outputs are read downstream. Any enclosing with block whose outputs are read downstream
    fails closed because context manager exit on premature termination or cleanup hazards
    cannot be assumed safe across arbitrary context managers (including locks, transactions,
    and streams)."""
    if scope_node is None or not outputs:
        return False
    try:
        for node in _get_scope_try_and_with_blocks(scope_node):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                with_body = getattr(node, "body", [])
                if _stmts_enclose_unit(with_body, u_start, u_end):
                    # Check if outputs are read downstream of the with block
                    # (where an exception-swallowing context manager resumes)
                    if _compound_stmt_reads_outputs_after(
                        scope_node, node, with_body, outputs
                    ):
                        return True
                continue

            if isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
                body_nodes = getattr(node, "body", [])
                orelse_nodes = getattr(node, "orelse", [])
                handlers = getattr(node, "handlers", [])
                finalbody_nodes = getattr(node, "finalbody", [])

                in_body = _stmts_enclose_unit(body_nodes, u_start, u_end)
                in_orelse = _stmts_enclose_unit(orelse_nodes, u_start, u_end)
                in_handler = any(
                    _stmts_enclose_unit(getattr(h, "body", []), u_start, u_end)
                    for h in handlers
                )

                if not (in_body or in_orelse or in_handler):
                    continue

                # In body, else, or except blocks, finally executes on abnormal exit (e.g. gen.close()).
                # If finally reads needed outputs, reject to prevent silent data loss or
                # UnboundLocalError.
                if _nodes_load_any_name(finalbody_nodes, outputs):
                    return True

                # If unit is inside try body, exceptions during execution jump to handlers.
                # If handlers read needed outputs, reject.
                if in_body:
                    handler_nodes: List[ast.AST] = []
                    for handler in handlers:
                        handler_nodes.extend(getattr(handler, "body", []))
                        if getattr(handler, "type", None):
                            handler_nodes.append(handler.type)
                    if _nodes_load_any_name(handler_nodes, outputs):
                        return True

                    # If any handler can fall through (e.g. swallows GeneratorExit or exceptions
                    # without re-raising or returning), check if any needed output is read
                    # downstream of the enclosing try block.
                    if handlers and any(
                        not _block_definitely_terminates(h.body) for h in handlers
                    ):
                        all_try_stmts = (
                            list(body_nodes)
                            + [s for h in handlers for s in getattr(h, "body", [])]
                            + list(orelse_nodes)
                            + list(finalbody_nodes)
                        )
                        if _compound_stmt_reads_outputs_after(
                            scope_node, node, all_try_stmts, outputs
                        ):
                            return True
        return False
    except (RecursionError, MemoryError):
        return True


def _merge_pre_unit_closure_reads(
    loaded: Set[str],
    pre_unit_captured_reads: Set[str],
    candidates: Optional[Set[str]],
) -> None:
    """Merges escaping pre-unit closure free reads into live downstream names fail-closed."""
    if candidates is not None:
        effective_captured = pre_unit_captured_reads & candidates
        if effective_captured:
            logger.debug(
                "Captured pre-unit closure candidate reads (fail-closed callback protection): %s",
                sorted(effective_captured),
            )
        loaded.update(effective_captured)
    else:
        if pre_unit_captured_reads:
            logger.debug(
                "Captured pre-unit closure reads (fail-closed callback protection): %s",
                sorted(pre_unit_captured_reads),
            )
        loaded.update(pre_unit_captured_reads)


def collect_downstream_read_names(
    source_text: str,
    unit: Dict[str, Any],
    candidates: Optional[Set[str]] = None,
    *,
    tree: Optional[ast.AST] = None,
    skip_pre_unit_closures: bool = False,
    source_digest: Optional[str] = None,
    unit_inputs: Optional[Set[str]] = None,
) -> Optional[Set[str]]:
    """Identifies variable names loaded downstream of a unit within its lexical execution scope.

    Module-Level Fallback:
    When a unit is defined at top-level module scope (having no enclosing FunctionDef),
    `scope_node` defaults to the module AST. All top-level function and class definitions
    prior to the unit in the module contribute their free global reads into the live candidate
    set. If an output candidate matches one of these captured names without a definite store
    beforehand, the generator subroutine will fail the definiteness check and fail closed
    (skipped during refactoring patch synthesis with a diagnostic DEBUG log).

    Args:
        source_text: The complete original Python source code.
        unit: AST unit mapping containing start and end coordinates.
        candidates: Optional set of variable names to restrict analysis to.
        tree: Optional pre-parsed module AST.
        skip_pre_unit_closures: Whether pre-unit closure read capture is bypassed.
        source_digest: Optional hex SHA-256 digest of source_text (must match
            hashlib.sha256(source_text.encode('utf-8', 'replace')).hexdigest()).
        unit_inputs: Precomputed live unit inputs.
    """
    # Invariant: downstream live names originate from three distinct sources:
    # (1) post-unit statements, (2) enclosing loop-carried paths, (3) escaping pre-unit closures.
    bounds = _get_valid_unit_bounds(unit)
    if bounds is None:
        return None
    u_start, u_end = bounds

    # Fast-path early out: if candidate set is explicitly empty, no outputs can match
    if candidates is not None and not candidates:
        return set()

    try:
        effective_tree = tree
        if effective_tree is None:
            effective_tree = _get_or_parse_tree(source_text, source_digest=source_digest)

        u_end_col = _extract_unit_end_col(unit)
        scope_node: Optional[ast.AST] = None
        if u_end_col is None:
            scope_node = _resolve_downstream_scope_node(source_text, unit, effective_tree)
            if scope_node is None:
                return None
            u_end_col = _resolve_unit_ast_end_col(scope_node, unit)

        effective_unit_inputs = unit_inputs
        if effective_unit_inputs is None and "inputs" in unit and unit["inputs"] is not None:
            coerced_inputs = _coerce_string_frozenset(unit["inputs"])
            effective_unit_inputs = (
                set(coerced_inputs) if coerced_inputs is not None else None
            )

        cache_key = _build_downstream_cache_key(
            source_text,
            unit,
            u_start,
            u_end,
            candidates,
            skip_pre_unit_closures,
            source_digest=source_digest,
            resolved_end_col=u_end_col,
            unit_inputs=effective_unit_inputs,
        )
        with _downstream_cache_lock:
            if cache_key in _downstream_reads_cache:
                cached = _downstream_reads_cache[cache_key]
                _downstream_reads_cache.move_to_end(cache_key)
                return set(cached)

        if scope_node is None:
            scope_node = _resolve_downstream_scope_node(source_text, unit, effective_tree)
            if scope_node is None:
                return None

        u_start_col = _extract_first_unit_coord(unit, ("start_col", "start_col_offset"))
        enclosing_loops = _find_enclosing_loops(
            scope_node,
            u_start,
            u_end,
            u_start_col=u_start_col,
        )

        if skip_pre_unit_closures:
            pre_unit_captured_reads: Set[str] = set()
            with _downstream_cache_lock:
                if not _warned_lenient_closure[0]:
                    _warned_lenient_closure[0] = True
                    logger.debug(
                        "Lenient closure mode active: suppressed pre-unit "
                        "closure read capture; escaping callback mutations "
                        "may be unobserved"
                    )
        else:
            pre_unit_captured_reads = _collect_pre_unit_closures(
                scope_node,
                u_start,
                candidates=candidates,
                u_start_col=u_start_col,
                module_tree=effective_tree,
            )

        is_module = isinstance(scope_node, ast.Module)
        outer_locals = _extract_scope_local_names(
            scope_node,
            effective_tree,
            include_classes=False,
            include_self_name=False,
        )

        visitor = _DownstreamReadVisitor(
            u_start,
            u_end,
            u_end_col,
            candidates=candidates,
            enclosing_loops=enclosing_loops,
            pass_mode=_VisitorPassMode.AFTER_UNIT,
            is_module_scope=is_module,
            u_start_col=u_start_col,
        )
        visitor.outer_bound.update(outer_locals)
        scope_stmts = _extract_scope_stmts(scope_node)
        for stmt in scope_stmts:
            visitor.visit(stmt)

        loaded = set(visitor.loaded)
        loop_visitor: Optional[_DownstreamReadVisitor] = None
        if enclosing_loops:
            loop_visitor = _DownstreamReadVisitor(
                u_start,
                u_end,
                u_end_col,
                candidates=candidates,
                enclosing_loops=enclosing_loops,
                pass_mode=_VisitorPassMode.LOOP_CARRIED,
                is_module_scope=is_module,
                u_start_col=u_start_col,
            )
            loop_visitor.outer_bound.update(outer_locals)
            for stmt in scope_stmts:
                loop_visitor.visit(stmt)
            loaded.update(loop_visitor.loaded)

            eff_inputs = (
                effective_unit_inputs
                if effective_unit_inputs is not None
                else _extract_unit_live_in_names(
                    scope_node,
                    u_start,
                    u_end,
                    u_end_col,
                    candidates=candidates,
                    u_start_col=u_start_col,
                )
            )
            cand_inputs = (
                set(candidates) & eff_inputs if candidates is not None else set(eff_inputs)
            )
            loop_killed = (
                loop_visitor.unit_entry_killed
                if loop_visitor.unit_entry_killed is not None
                else set()
            )
            loaded.update(cand_inputs - loop_killed)

        has_dynamic = visitor.has_dynamic_read or (
            loop_visitor is not None and loop_visitor.has_dynamic_read
        )
        if has_dynamic:
            if candidates is not None:
                loaded.update(candidates)
            else:
                for subnode in ast.walk(scope_node):
                    if isinstance(subnode, ast.Name) and isinstance(
                        subnode.ctx, (ast.Store, ast.Load)
                    ):
                        if subnode.id not in _BUILTIN_NAMES and not subnode.id.startswith("__"):
                            loaded.add(subnode.id)

        _merge_pre_unit_closure_reads(loaded, pre_unit_captured_reads, candidates)

        _store_lru_cache_entry(
            _downstream_reads_cache,
            cache_key,
            frozenset(loaded),
            _MAX_DOWNSTREAM_CACHE_SIZE,
        )

        return loaded
    except (RecursionError, MemoryError):
        return None


def has_async_generator_delegation_hazard(
    *scopes: Optional[Dict[str, Any]],
    has_outputs: bool = False,
) -> bool:
    """Checks whether scopes represent an async generator that cannot safely be delegated
    via PEP 525 async for ...: yield delegation without semantic loss (including return values,
    unpaired outputs, bidirectional yield assignments, or try/with/async with cleanup blocks
    that lose athrow/aclose propagation).

    Applies to both subroutine and whole-function async generator refactorings.
    """
    has_async_gen = any(
        bool(s.get("has_yield")) and bool(s.get("is_async"))
        for s in scopes
        if s is not None
    )
    if not has_async_gen:
        return False
    has_ret = has_outputs or any(
        bool(s.get("has_return_value")) for s in scopes if s is not None
    )
    if has_ret:
        return True
    if any(bool(s.get("has_yield_assignment")) for s in scopes if s is not None):
        return True
    if any(bool(s.get("has_try_or_with")) for s in scopes if s is not None):
        return True
    return False


is_async_generator_with_return_value = has_async_generator_delegation_hazard
