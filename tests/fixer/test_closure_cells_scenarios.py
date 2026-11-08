"""Parametrized closure-cell scenarios, integration tests, and differential refactoring cases."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from pydoppelgangerhunt.fixer import analyze_unit_variable_scope
from pydoppelgangerhunt.fixer.dataflow import collect_downstream_read_names
from tests.fixer._closure_test_helpers import _run_differential_refactoring_case


_CLOSURE_CELL_SCENARIOS = [
    # --- Positive cases (hazard raised) ---
    (
        "post_unit_mutated_cell",
        (
            "def build_post(items):\n"
            "    scale = 1\n"
            "    callbacks = []\n"
            "    for item in items:\n"
            "        callbacks.append(lambda x, item=item: x * scale + item)\n"
            "    scale = 10\n"
            "    return callbacks\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    callbacks.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "loop_mutated_cell",
        (
            "def build_loop(items):\n"
            "    callbacks = []\n"
            "    for item in items:\n"
            "        def cb(x):\n"
            "            return x + item\n"
            "        callbacks.append(cb)\n"
            "    return callbacks\n"
        ),
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_code": "def cb(x):\n    return x + item\ncallbacks.append(cb)\n",
        },
        True,
    ),
    (
        "intra_unit_nonlocal_mutator_with_escaping_cell",
        (
            "def outer_inc():\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    inc()\n"
            "    return count\n"
        ),
        {
            "start": 2,
            "end": 5,
            "kind": "compound_block",
            "source_code": "count = 0\ndef inc():\n    nonlocal count\n    count += 1\n",
        },
        True,
    ),
    (
        "pre_unit_sibling_closure_sharing_mutated_cell",
        (
            "def outer_sib(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(get_acc())\n"
            "    return acc, out\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    out.append(get_acc())\n",
        },
        True,
    ),
    (
        "pre_unit_closure_mutating_nonlocal_cell_passed_to_unit",
        (
            "def outer_nl_sib(items):\n"
            "    acc = 0\n"
            "    def bump():\n"
            "        nonlocal acc\n"
            "        acc += 1\n"
            "    for x in items:\n"
            "        bump()\n"
            "        use(acc + x)\n"
            "    return acc\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": "for x in items:\n    bump()\n    use(acc + x)\n",
        },
        True,
    ),
    (
        "multilevel_intermediate_nonlocal_mutation",
        (
            "def outer_inter(items):\n"
            "    scale = 1\n"
            "    def inner_inter():\n"
            "        nonlocal scale\n"
            "        cbs = []\n"
            "        for item in items:\n"
            "            cbs.append(lambda x, item=item: x * scale + item)\n"
            "        scale = 20\n"
            "        return cbs\n"
            "    return inner_inter\n"
        ),
        {
            "start": 6,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "whole_closure_post_unit_cell_mutation",
        (
            "def make_handlers(items):\n"
            "    scale = 1\n"
            "    def handle(x):\n"
            "        total = 0\n"
            "        for item in items:\n"
            "            total += (x + item) * scale\n"
            "        return total\n"
            "    scale = 10\n"
            "    return handle\n"
        ),
        {
            "start": 3,
            "end": 7,
            "kind": "closure",
            "is_subroutine": False,
            "source_code": (
                "def handle(x):\n"
                "    total = 0\n"
                "    for item in items:\n"
                "        total += (x + item) * scale\n"
                "    return total\n"
            ),
        },
        True,
    ),
    (
        "whole_closure_nonlocal_store",
        (
            "def make_counter():\n"
            "    count = 0\n"
            "    def step(delta):\n"
            "        nonlocal count\n"
            "        count += delta\n"
            "        return count\n"
            "    return step\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "closure",
            "is_subroutine": False,
            "source_code": (
                "def step(delta):\n"
                "    nonlocal count\n"
                "    count += delta\n"
                "    return count\n"
            ),
        },
        True,
    ),
    (
        "assigned_lambda_sibling_closure_in_check4",
        (
            "def outer_lam(items):\n"
            "    bias = 0\n"
            "    get_bias = lambda: bias\n"
            "    out = []\n"
            "    for x in items:\n"
            "        bias += x\n"
            "        out.append(get_bias())\n"
            "    return bias, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": "for x in items:\n    bias += x\n    out.append(get_bias())\n",
        },
        True,
    ),
    (
        "intra_unit_mutator_escaping_via_container_to_pre_unit_reader",
        (
            "def outer_escape(callbacks):\n"
            "    def get_count():\n"
            "        return count\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    callbacks.append(inc)\n"
            "    return get_count\n"
        ),
        {
            "start": 4,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "def inc():\n"
                "    nonlocal count\n"
                "    count += 1\n"
                "callbacks.append(inc)\n"
            ),
        },
        True,
    ),
    (
        "classdef_namespace_isolation_shadowing",
        (
            "def outer_cls(items):\n"
            "    class C:\n"
            "        scale = 1\n"
            "        def m(self):\n"
            "            cbs = []\n"
            "            for item in items:\n"
            "                cbs.append(lambda x, item=item: x * scale + item)\n"
            "            return cbs\n"
            "    scale = 10\n"
            "    return C\n"
        ),
        {
            "start": 6,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "pre_bound_outer_var_mutated_by_intra_unit_closure",
        (
            "def outer_inc():\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    inc()\n"
            "    return count\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": "def inc():\n    nonlocal count\n    count += 1\ninc()\n",
        },
        True,
    ),
    (
        "nested_closure_unit_with_inner_nonlocal_mutation",
        (
            "def outer_mid():\n"
            "    count = 0\n"
            "    def middle():\n"
            "        def inner():\n"
            "            nonlocal count\n"
            "            count += 1\n"
            "        inner()\n"
            "    return middle, count\n"
        ),
        {
            "start": 3,
            "end": 7,
            "kind": "closure",
            "is_subroutine": False,
            "source_code": (
                "def middle():\n"
                "    def inner():\n"
                "        nonlocal count\n"
                "        count += 1\n"
                "    inner()\n"
            ),
        },
        True,
    ),
    (
        "pre_unit_anon_lambda_and_decorator_escape",
        (
            "def outer_anon(callbacks, readers):\n"
            "    readers.append(lambda: count)\n"
            "    count = 0\n"
            "    @callbacks.append\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
        ),
        {
            "start": 3,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "@callbacks.append\n"
                "def inc():\n"
                "    nonlocal count\n"
                "    count += 1\n"
            ),
        },
        True,
    ),
    (
        "conditional_delete_in_enclosing_if",
        (
            "def build_cdel(items, cond):\n"
            "    scale = 1\n"
            "    if cond:\n"
            "        del scale\n"
            "        for item in items:\n"
            "            cbs.append(lambda x, item=item: x * scale + item)\n"
        ),
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "conditional_store_before_span_in_if",
        (
            "def build_cstore(items, cond):\n"
            "    if cond:\n"
            "        scale = 1\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "conditional_store_before_span_in_suppressing_with",
        (
            "def build_with_suppress(items, d):\n"
            "    with suppress(KeyError):\n"
            "        scale = d['x']\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "transitive_sibling_closure_calls",
        (
            "def outer_trans(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    def step():\n"
            "        return get_acc() * 2\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(step())\n"
            "    return acc, out\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    out.append(step())\n",
        },
        True,
    ),
    (
        "lazy_genexp_walrus_mutator",
        (
            "def run_genexp_walrus(items):\n"
            "    acc = 0\n"
            "    if items:\n"
            "        gen = ((acc := acc + x) for x in items)\n"
            "    else:\n"
            "        gen = iter(())\n"
            "    list(gen)\n"
            "    return acc\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    gen = ((acc := acc + x) for x in items)\n"
                "else:\n"
                "    gen = iter(())\n"
            ),
        },
        True,
    ),
    (
        "except_as_target_implicit_exit_delete",
        (
            "def run_except_as(items, fn):\n"
            "    cbs = []\n"
            "    try:\n"
            "        fn()\n"
            "    except ValueError as exc:\n"
            "        for x in items:\n"
            "            cbs.append(lambda x=x: (x, exc))\n"
            "    return cbs\n"
        ),
        {
            "start": 6,
            "end": 7,
            "kind": "compound_block",
            "source_code": "for x in items:\n    cbs.append(lambda x=x: (x, exc))\n",
        },
        True,
    ),
    (
        "self_name_rebind_in_outer_scope",
        (
            "def outer_rebind_fn(items):\n"
            "    def scale():\n"
            "        cbs = []\n"
            "        for x in items:\n"
            "            cbs.append(lambda x=x: (x, scale))\n"
            "        return cbs\n"
            "    res = scale()\n"
            "    scale = 10\n"
            "    return res\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": "for x in items:\n    cbs.append(lambda x=x: (x, scale))\n",
        },
        True,
    ),
    (
        "async_coroutine_nonlocal_mutation_across_await",
        (
            "def outer_async(items, checkpoint):\n"
            "    total = 0\n"
            "    async def worker():\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            total += x\n"
            "            await checkpoint()\n"
            "    return worker, lambda: total\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": "for x in items:\n    total += x\n    await checkpoint()\n",
        },
        True,
    ),
    (
        "readonly_generator_capturing_mutable_outer_cell",
        (
            "def outer_ro_gen(items):\n"
            "    scale = 2\n"
            "    def set_scale(v):\n"
            "        nonlocal scale\n"
            "        scale = v\n"
            "    def gen():\n"
            "        for x in items:\n"
            "            yield scale * x\n"
            "    return gen, set_scale\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": "for x in items:\n    yield scale * x\n",
        },
        True,
    ),
    (
        "aliased_sibling_closure_callback",
        (
            "def outer_alias_container(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    reader = get_acc\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(reader())\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    out.append(reader())\n",
        },
        True,
    ),
    (
        "container_stored_sibling_closure_callback",
        (
            "def outer_container_cb(items):\n"
            "    acc = 0\n"
            "    readers = [lambda: acc]\n"
            "    out = []\n"
            "    for y in items:\n"
            "        acc += y\n"
            "        out.append(readers[0]())\n"
            "    return acc, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": "for y in items:\n    acc += y\n    out.append(readers[0]())\n",
        },
        True,
    ),
    (
        "nested_def_default_walrus_in_closure",
        (
            "def outer_def_walrus(items):\n"
            "    acc = 0\n"
            "    def make_fn():\n"
            "        nonlocal acc\n"
            "        def inner(val=(acc := acc + 1)):\n"
            "            return val\n"
            "        return inner\n"
            "    out = []\n"
            "    for x in items:\n"
            "        fn = make_fn()\n"
            "        out.append(fn() + acc + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    fn = make_fn()\n"
                "    out.append(fn() + acc + x)\n"
            ),
        },
        True,
    ),
    # --- Negative cases (no hazard raised) ---
    (
        "neg_closure_capturing_never_reassigned_parameter",
        (
            "def build_param_only(items, factor):\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v: v * factor)\n"
            "    return cbs\n"
        ),
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_code": "for x in items:\n    cbs.append(lambda v: v * factor)\n",
        },
        False,
    ),
    (
        "neg_item_default_binding_with_constant_cell",
        (
            "def build_safe(items, scale):\n"
            "    callbacks = []\n"
            "    for item in items:\n"
            "        callbacks.append(lambda x, item=item: x * scale + item)\n"
            "    return callbacks\n"
        ),
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    callbacks.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "neg_nonlocal_mutation_in_unit_with_no_escaping_readers",
        (
            "def compute_local_nl(items):\n"
            "    if items:\n"
            "        count = 0\n"
            "        def bump(step):\n"
            "            nonlocal count\n"
            "            count += step\n"
            "        for x in items:\n"
            "            bump(x)\n"
            "    return len(items)\n"
        ),
        {
            "start": 2,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    count = 0\n"
                "    def bump(step):\n"
                "        nonlocal count\n"
                "        count += step\n"
                "    for x in items:\n"
                "        bump(x)\n"
            ),
        },
        False,
    ),
    (
        "neg_eagerly_consumed_genexp_capturing_later_mutated_var",
        (
            "def compute_eager_genexp(items):\n"
            "    scale = 2\n"
            "    if items:\n"
            "        total = sum(v * scale for v in items)\n"
            "    else:\n"
            "        total = 0\n"
            "    scale = 10\n"
            "    return total + scale\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    total = sum(v * scale for v in items)\n"
                "else:\n"
                "    total = 0\n"
            ),
        },
        False,
    ),
    (
        "neg_eagerly_consumed_genexp_walrus_not_lazy_mutator",
        (
            "def compute_eager_walrus(items):\n"
            "    acc = 0\n"
            "    if items:\n"
            "        total = sum((acc := acc + v) for v in items)\n"
            "    else:\n"
            "        total = 0\n"
            "    return total + acc\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    total = sum((acc := acc + v) for v in items)\n"
                "else:\n"
                "    total = 0\n"
            ),
        },
        False,
    ),
    (
        "neg_call_result_passed_to_container_append_not_closure_alias",
        (
            "def run_no_alias(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    out = []\n"
            "    out.append(get_acc())\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    out.append(x)\n",
        },
        False,
    ),
    (
        "neg_binding_in_same_enclosing_with_block_is_pre_bound",
        (
            "def build_in_same_with(items, ctx):\n"
            "    cbs = []\n"
            "    with ctx:\n"
            "        scale = 2\n"
            "        for item in items:\n"
            "            cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "neg_lazy_genexp_first_iterable_evaluated_eagerly",
        (
            "def build_gen_first_iter(items):\n"
            "    seq = list(items)\n"
            "    if seq:\n"
            "        gen = (x * 2 for x in seq)\n"
            "    else:\n"
            "        gen = iter(())\n"
            "    seq = []\n"
            "    return list(gen)\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if seq:\n"
                "    gen = (x * 2 for x in seq)\n"
                "else:\n"
                "    gen = iter(())\n"
            ),
        },
        False,
    ),
    (
        "multilevel_intermediate_pre_unit_nonlocal_mutation",
        (
            "def outer_inter_pre(items):\n"
            "    scale = 1\n"
            "    def inner_inter():\n"
            "        nonlocal scale\n"
            "        scale += 1\n"
            "        cbs = []\n"
            "        for item in items:\n"
            "            cbs.append(lambda x, item=item: x * scale + item)\n"
            "        return cbs\n"
            "    return inner_inter\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "pre_unit_closure_factory_assigned_to_variable",
        (
            "def outer_factory(items):\n"
            "    acc = 0\n"
            "    def make_closure():\n"
            "        def read_acc():\n"
            "            return acc\n"
            "        return read_acc\n"
            "    fn = make_closure()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(fn() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(fn() + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_exhaustive_if_else_branch_initialization_before_unit",
        (
            "def build_if_else_init(items, cond):\n"
            "    cbs = []\n"
            "    if cond:\n"
            "        scale = 1\n"
            "    else:\n"
            "        scale = 2\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "neg_enclosing_local_class_method_synchronous_nonlocal_mutation",
        (
            "def outer_cls_method(log):\n"
            "    total = 0\n"
            "    class Worker:\n"
            "        def run(self, items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                total += x * 2\n"
            "                log(x)\n"
            "    return Worker\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x * 2\n"
                "    log(x)\n"
            ),
        },
        False,
    ),
    (
        "outer_global_captured_by_intra_unit_closure",
        (
            "def outer_glob_cap(items):\n"
            "    global shared_g\n"
            "    shared_g = 1\n"
            "    out = []\n"
            "    for x in items:\n"
            "        out.append(lambda y, x=x: y + shared_g + x)\n"
            "    shared_g = 2\n"
            "    return out\n"
        ),
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    out.append(lambda y, x=x: y + shared_g + x)\n"
            ),
        },
        True,
    ),
    (
        "multilevel_outer_loop_escaping_nonlocal_unit_mutation",
        (
            "def outer_loop_nl(groups):\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    for grp in groups:\n"
            "        def inner_step(x):\n"
            "            nonlocal total\n"
            "            if x > 0:\n"
            "                total += x\n"
            "                cbs.append(lambda y: y + total + x)\n"
            "        inner_step(grp)\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "if x > 0:\n"
                "    total += x\n"
                "    cbs.append(lambda y: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "eager_comprehension_aliased_sibling_closure",
        (
            "def outer_comp_alias(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    readers = [get_acc for _ in range(3)]\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(readers[0]() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(readers[0]() + x)\n"
            ),
        },
        True,
    ),
    (
        "nested_class_body_nonlocal_unbound_conditional_mutation",
        (
            "def outer_cls_nl(items):\n"
            "    def mid():\n"
            "        class Box:\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                if x > 0:\n"
            "                    total = x * 2\n"
            "        return Box\n"
            "    total = 0\n"
            "    return mid\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    if x > 0:\n"
                "        total = x * 2\n"
            ),
        },
        True,
    ),
    (
        "generator_mutating_local_cell_captured_by_sibling_closure",
        (
            "def outer_gen_sib(items, callbacks):\n"
            "    acc = 0\n"
            "    callbacks.append(lambda: acc)\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        yield x\n"
        ),
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    yield x\n",
        },
        True,
    ),
    (
        "multicall_nested_fn_mutating_and_capturing_nonlocal_cell",
        (
            "def outer_mc():\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    def inner_step(items):\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            total += x\n"
            "            cbs.append(lambda y, x=x: y + total + x)\n"
            "    inner_step([1])\n"
            "    inner_step([2])\n"
            "    return cbs\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_exhaustive_if_else_after_del_rebinds_before_unit",
        (
            "def build_del_rebind(items, cond):\n"
            "    scale = 0\n"
            "    cbs = []\n"
            "    if cond:\n"
            "        del scale\n"
            "        scale = 1\n"
            "    else:\n"
            "        scale = 2\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "exhaustive_if_else_with_nested_single_branch_del_not_definite",
        (
            "def build_nested_del(items, cond, sub):\n"
            "    cbs = []\n"
            "    if cond:\n"
            "        scale = 1\n"
            "        if sub:\n"
            "            del scale\n"
            "    else:\n"
            "        scale = 2\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "closure_factory_returning_sibling_closure_and_pop",
        (
            "def outer_ret_sib(items):\n"
            "    acc = 0\n"
            "    def read_acc():\n"
            "        return acc\n"
            "    def get_reader():\n"
            "        return read_acc\n"
            "    cbs = [get_reader()]\n"
            "    fn = cbs.pop()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(fn() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 10,
            "end": 12,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(fn() + x)\n"
            ),
        },
        True,
    ),
    (
        "generator_factory_assigned_to_variable",
        (
            "def outer_gen_fac(items):\n"
            "    acc = 0\n"
            "    def make_gen():\n"
            "        yield acc\n"
            "    g = make_gen()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(next(g) + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(next(g) + x)\n"
            ),
        },
        True,
    ),
    (
        "intra_unit_mutator_self_registering_in_body",
        (
            "def outer_self_reg(callbacks):\n"
            "    def get_count():\n"
            "        return count\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "        callbacks.append(inc)\n"
            "    inc()\n"
            "    return get_count\n"
        ),
        {
            "start": 4,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "def inc():\n"
                "    nonlocal count\n"
                "    count += 1\n"
                "    callbacks.append(inc)\n"
                "inc()\n"
            ),
        },
        True,
    ),
    (
        "intra_unit_nested_factory_returning_mutator",
        (
            "def outer_nested_fac(callbacks):\n"
            "    def get_count():\n"
            "        return count\n"
            "    count = 0\n"
            "    def make_inc():\n"
            "        def inc():\n"
            "            nonlocal count\n"
            "            count += 1\n"
            "        return inc\n"
            "    callbacks.append(make_inc())\n"
            "    return get_count\n"
        ),
        {
            "start": 4,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "def make_inc():\n"
                "    def inc():\n"
                "        nonlocal count\n"
                "        count += 1\n"
                "    return inc\n"
                "callbacks.append(make_inc())\n"
            ),
        },
        True,
    ),
    (
        "neg_sync_subroutine_nonlocal_mut_with_post_def_outer_binding",
        (
            "def outer_post_def(log):\n"
            "    total = 0\n"
            "    def worker(items):\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            total += x * 2\n"
            "            log(x)\n"
            "    worker([1, 2])\n"
            "    total += 1\n"
            "    return total\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x * 2\n"
                "    log(x)\n"
            ),
        },
        False,
    ),
    (
        "neg_len_of_closure_container_is_not_closure_alias",
        (
            "def outer_len_cbs(items):\n"
            "    acc = 0\n"
            "    callbacks = [lambda: acc]\n"
            "    n = len(callbacks)\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(n + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(n + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_method_inside_class_does_not_inherit_class_body_nonlocal",
        (
            "def outer_cls_method_iso(items, log):\n"
            "    total = 0\n"
            "    class Box:\n"
            "        nonlocal total\n"
            "        total = 1\n"
            "        def run(self):\n"
            "            total = 0\n"
            "            for x in items:\n"
            "                total += x\n"
            "                log(x)\n"
            "            return total\n"
            "    return Box, total\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x\n"
                "    log(x)\n"
            ),
        },
        False,
    ),
    (
        "eager_listcomp_walrus_body_is_conditional_not_pre_bound",
        (
            "def build_comp_walrus(items, empty_seq):\n"
            "    cbs = []\n"
            "    _ = [(scale := v) for v in empty_seq]\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "neg_lazy_genexp_first_iterable_walrus_is_eager_not_lazy_mutator",
        (
            "def build_gen_hdr_walrus(items):\n"
            "    if items:\n"
            "        gen = ((seq := list(items)), (x * 2 for x in seq))[1]\n"
            "    else:\n"
            "        seq = []\n"
            "        gen = iter(())\n"
            "    return list(gen), seq\n"
        ),
        {
            "start": 2,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    gen = ((seq := list(items)), (x * 2 for x in seq))[1]\n"
                "else:\n"
                "    seq = []\n"
                "    gen = iter(())\n"
            ),
        },
        False,
    ),
    (
        "comprehension_append_sibling_closure_carrier",
        (
            "def outer_comp_append(items):\n"
            "    acc = 0\n"
            "    readers = []\n"
            "    _ = [readers.append(lambda: acc) for _ in range(2)]\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(readers[0]() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(readers[0]() + x)\n"
            ),
        },
        True,
    ),
    (
        "multilevel_sibling_closure_nonlocal_deletion_in_inner_scope",
        (
            "def outer_ml_del(items):\n"
            "    total = 5\n"
            "    def inner():\n"
            "        def clear():\n"
            "            nonlocal total\n"
            "            del total\n"
            "        clear()\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total = x * 2\n"
            "    return inner\n"
        ),
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    if x > 0:\n"
                "        total = x * 2\n"
            ),
        },
        True,
    ),
    (
        "conditional_augassign_on_unbound_nonlocal_cell",
        (
            "def outer_unb_aug():\n"
            "    def inner_unb(items):\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total += x * 2\n"
            "    inner_unb([-1, -2])\n"
            "    total = 0\n"
            "    return inner_unb\n"
        ),
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    if x > 0:\n"
                "        total += x * 2\n"
            ),
        },
        True,
    ),
    (
        "neg_len_of_inline_lambda_comprehension_is_not_closure_alias",
        (
            "def outer_len_inline_comp(items):\n"
            "    acc = 0\n"
            "    n = len([lambda: acc for _ in range(3)])\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(n + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(n + x)\n"
            ),
        },
        False,
    ),
    (
        "generator_loop_initialized_cell_captured_by_pre_unit_sibling_closure",
        (
            "def outer_gen_loop_init(items, callbacks):\n"
            "    callbacks.append(lambda: acc)\n"
            "    for x in items:\n"
            "        acc = x * 2\n"
            "        yield x\n"
        ),
        {
            "start": 3,
            "end": 5,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc = x * 2\n    yield x\n",
        },
        True,
    ),
    (
        "multicall_enclosed_fn_invoked_in_comprehension",
        (
            "def outer_comp_call(groups):\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    def inner_step(items):\n"
            "        nonlocal total\n"
            "        total += 1\n"
            "        for x in items:\n"
            "            cbs.append(lambda y, x=x: y + total + x)\n"
            "    _ = [inner_step(g) for g in groups]\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "multicall_enclosed_fn_escaping_via_return",
        (
            "def outer_ret_fn():\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    def inner_step(items):\n"
            "        nonlocal total\n"
            "        total += 1\n"
            "        for x in items:\n"
            "            cbs.append(lambda y, x=x: y + total + x)\n"
            "    return inner_step, cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "lazy_genexp_first_iterable_closure_carrier_alias",
        (
            "def outer_lazy_hdr_carrier(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    gen = (f() for f in [get_acc])\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(next(gen) + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(next(gen) + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_helper_with_eager_sum_genexp_is_not_closure_factory",
        (
            "def outer_sum_gen(items):\n"
            "    acc = 0\n"
            "    def compute_initial():\n"
            "        return sum(x * acc for x in items)\n"
            "    init_val = compute_initial()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(init_val + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(init_val + x)\n"
            ),
        },
        False,
    ),
    (
        "conditional_walrus_in_comprehension_first_iterable_not_pre_bound",
        (
            "def build_nested_hdr_walrus(items, empty_seq):\n"
            "    cbs = []\n"
            "    _ = [(scale := v) for x in [1] for v in empty_seq]\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "neg_comprehension_target_shadowing_outer_cell_in_lambda",
        (
            "def outer_shadow_target(items):\n"
            "    acc = [1, 2]\n"
            "    cbs = [lambda: acc for acc in acc]\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc = [x]\n"
            "        out.append(cbs[0]() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc = [x]\n"
                "    out.append(cbs[0]() + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_pre_bound_var_deleted_and_rebound_in_one_if_branch",
        (
            "def build_del_rebind_branch(items, cond):\n"
            "    scale = 1\n"
            "    if cond:\n"
            "        del scale\n"
            "        scale = 2\n"
            "    else:\n"
            "        pass\n"
            "    cbs = []\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "neg_multilevel_shadowed_sibling_closure_cell_no_sharing",
        (
            "def outer_shadow_sib(items):\n"
            "    acc = 100\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    def inner():\n"
            "        acc = 0\n"
            "        out = []\n"
            "        for x in items:\n"
            "            acc += x\n"
            "            out.append(get_acc() + acc)\n"
            "        return acc, out\n"
            "    return inner\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(get_acc() + acc)\n"
            ),
        },
        False,
    ),
    (
        "multicall_enclosed_fn_defined_in_local_class",
        (
            "def outer_cls_method(groups):\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    class Runner:\n"
            "        def step(self, items):\n"
            "            nonlocal total\n"
            "            total += 1\n"
            "            for x in items:\n"
            "                cbs.append(lambda y, x=x: y + total + x)\n"
            "    return Runner, cbs\n"
        ),
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "class_body_mutating_carrier_call_captures_shared_cell",
        (
            "def outer_cls_carrier(items):\n"
            "    acc = 0\n"
            "    cbs = []\n"
            "    class Box:\n"
            "        cbs.append(lambda: acc)\n"
            "        out = []\n"
            "        for x in items:\n"
            "            acc += x\n"
            "            out.append(cbs[0]() + x)\n"
            "    return Box, acc\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(cbs[0]() + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_sibling_class_shadowed_attribute_not_post_read",
        (
            "def outer_sib_cls_attr(items):\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    for x in items:\n"
            "        inc()\n"
            "    class Sibling:\n"
            "        count = 10\n"
            "        doubled = count * 2\n"
            "    return Sibling\n"
        ),
        {
            "start": 2,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "def inc():\n"
                "    nonlocal count\n"
                "    count += 1\n"
                "for x in items:\n"
                "    inc()\n"
            ),
        },
        False,
    ),
    (
        "multicall_enclosed_fn_self_escaping_in_own_body",
        (
            "def outer_self_rec(callbacks):\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    def inner_step(items):\n"
            "        nonlocal total\n"
            "        total += 1\n"
            "        for x in items:\n"
            "            cbs.append(lambda y, x=x: y + total + x)\n"
            "        callbacks.append(inner_step)\n"
            "    inner_step([1])\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "subroutine_unit_containing_direct_nonlocal_statement",
        (
            "def outer_direct_nl(items):\n"
            "    total = 0\n"
            "    def inner():\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            total += x\n"
            "    return inner, total\n"
        ),
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "nonlocal total\n"
                "for x in items:\n"
                "    total += x\n"
            ),
        },
        True,
    ),
    (
        "try_except_earlier_handler_binding_not_definite_in_later_handler",
        (
            "def outer(items):\n"
            "    try:\n"
            "        perform_operation()\n"
            "    except ValueError:\n"
            "        scale = 10\n"
            "    except TypeError:\n"
            "        for x in items:\n"
            "            callback = lambda: scale\n"
            "        return callback\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    callback = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "try_except_in_loop_earlier_handler_binding_captured_in_later_handler",
        (
            "def outer(batches, items):\n"
            "    for batch in batches:\n"
            "        try:\n"
            "            perform_operation(batch)\n"
            "        except ValueError:\n"
            "            scale = 10\n"
            "        except TypeError:\n"
            "            for x in items:\n"
            "                callback = lambda: scale\n"
            "            return callback\n"
        ),
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    callback = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "try_except_handler_binding_not_definite_in_else_block",
        (
            "def outer(items):\n"
            "    try:\n"
            "        perform_operation()\n"
            "    except ValueError:\n"
            "        scale = 10\n"
            "    else:\n"
            "        for x in items:\n"
            "            callback = lambda: scale\n"
            "        return callback\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    callback = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "try_except_handler_and_else_binding_not_definite_in_finally_block",
        (
            "def outer(items):\n"
            "    try:\n"
            "        perform_operation()\n"
            "    except ValueError:\n"
            "        scale = 10\n"
            "    else:\n"
            "        scale = 20\n"
            "    finally:\n"
            "        for x in items:\n"
            "            callback = lambda: scale\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    callback = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "neg_try_except_own_handler_target_and_prior_stmt_definitely_bound",
        (
            "def outer(items):\n"
            "    try:\n"
            "        perform_operation()\n"
            "    except ValueError:\n"
            "        other = 1\n"
            "    except TypeError as err:\n"
            "        scale = 10\n"
            "        for x in items:\n"
            "            msg = str(err)\n"
            "            callback = lambda: scale\n"
            "        return callback, msg\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    msg = str(err)\n"
                "    callback = lambda: scale\n"
            ),
        },
        False,
    ),
    (
        "builtin_call_arg_invoking_closure_factory_and_mutating_outer_container",
        (
            "def outer(items):\n"
            "    value = 1\n"
            "    saved = []\n"
            "    def make_callback():\n"
            "        callback = lambda: value\n"
            "        saved.append(callback)\n"
            "        return callback\n"
            "    result = len([make_callback()])\n"
            "    for x in items:\n"
            "        value = x + result\n"
            "    return saved, value\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    value = x + result\n"
            ),
        },
        True,
    ),
    (
        "builtin_call_arg_outer_container_read_by_unit",
        (
            "def outer(items):\n"
            "    value = 1\n"
            "    saved = []\n"
            "    def make_callback():\n"
            "        callback = lambda: value\n"
            "        saved.append(callback)\n"
            "        return callback\n"
            "    result = len([make_callback()])\n"
            "    for x in items:\n"
            "        value = x + len(saved)\n"
            "    return result, value\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    value = x + len(saved)\n"
            ),
        },
        True,
    ),
    (
        "multiline_decorator_unit_capturing_uninitialized_decorated_fn",
        (
            "def outer(items, register):\n"
            "    @register(\n"
            "        items,\n"
            "        [lambda x: x * inner() for _ in items],\n"
            "    )\n"
            "    def inner():\n"
            "        return 2\n"
            "    return inner\n"
        ),
        {
            "start": 4,
            "end": 4,
            "start_col": 8,
            "end_col": 46,
            "kind": "comprehension",
            "source_code": "[lambda x: x * inner() for _ in items]",
        },
        True,
    ),
    (
        "same_line_decorator_unit_capturing_late_bound_variable",
        (
            "def outer(items, dec):\n"
            "    @dec([lambda x: x * scale for _ in items])\n"
            "    def inner():\n"
            "        return scale\n"
            "    scale = 10\n"
            "    return inner\n"
        ),
        {
            "start": 2,
            "end": 2,
            "start_col": 9,
            "end_col": 45,
            "kind": "comprehension",
            "source_code": "[lambda x: x * scale for _ in items]",
        },
        True,
    ),
    (
        "nested_decorated_defs_unit_inside_inner_body_captures_outer_late_var",
        (
            "def outer(items, dec1, dec2):\n"
            "    @dec1(\n"
            "        1,\n"
            "    )\n"
            "    @dec2\n"
            "    def mid():\n"
            "        for x in items:\n"
            "            cb = lambda: scale\n"
            "        return cb\n"
            "    scale = 10\n"
            "    return mid\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cb = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "decorated_first_stmt_closure_unit_captures_late_bound_var",
        (
            "def o(d):\n"
            "    @d\n"
            "    def i():\n"
            "        return scale\n"
            "    scale = 1\n"
            "    return i\n"
        ),
        {
            "start": 2,
            "end": 4,
            "start_col": 4,
            "kind": "closure",
            "source_code": (
                "@d\n"
                "def i():\n"
                "    return scale\n"
            ),
        },
        True,
    ),
    (
        "enumerate_carrier_closure_reads_unit_mutated_var",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    pairs = enumerate([lambda: acc])\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(next(pairs)[1]())\n"
            "    return acc, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(next(pairs)[1]())\n"
            ),
        },
        True,
    ),
    (
        "enclosing_local_class_method_calls_sibling_peek_reading_nonlocal",
        (
            "def outer(items, log):\n"
            "    total = 0\n"
            "    class Worker:\n"
            "        def peek(self):\n"
            "            return total\n"
            "        def run(self, items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                total += x * 2\n"
            "                log(self.peek())\n"
            "    return Worker\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x * 2\n"
                "    log(self.peek())\n"
            ),
        },
        True,
    ),
    (
        "neg_with_as_target_definitely_pre_bound_before_unit",
        (
            "def outer(ctx, items):\n"
            "    with ctx() as fh:\n"
            "        pass\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda x=x: (fh, x))\n"
            "    return cbs\n"
        ),
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda x=x: (fh, x))\n"
            ),
        },
        False,
    ),
    (
        "neg_if_return_else_fallthrough_binding_before_unit",
        (
            "def outer(cond, items):\n"
            "    if cond:\n"
            "        y = 1\n"
            "        return []\n"
            "    else:\n"
            "        scale = 2\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v, x=x: v * scale + x)\n"
            "    return cbs\n"
        ),
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda v, x=x: v * scale + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_inner_fn_mutating_own_local_container_not_outer_carrier",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    holder = []\n"
            "    def helper():\n"
            "        holder = []\n"
            "        holder.append(lambda: acc)\n"
            "        return len(holder)\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        holder.append(acc)\n"
            "    return acc, holder\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    holder.append(acc)\n"
            ),
        },
        False,
    ),
    (
        "neg_factory_with_bare_annotation_shadows_outer_closure_name",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    cb = lambda: acc\n"
            "    def get_val():\n"
            "        cb: object\n"
            "        return 1\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(get_val())\n"
            "    return acc, out, cb\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(get_val())\n"
            ),
        },
        False,
    ),
    (
        "neg_same_line_pre_bound_with_none_start_col_inferred",
        (
            "def outer(items):\n"
            "    scale = 10; cbs = [lambda x: x * scale for _ in items]\n"
            "    return cbs\n"
        ),
        {
            "start": 2,
            "end": 2,
            "kind": "comprehension",
            "source_code": "[lambda x: x * scale for _ in items]",
        },
        False,
    ),
    (
        "same_line_post_mutated_with_none_start_col_inferred",
        (
            "def outer(items):\n"
            "    cbs = [lambda x: x * scale for _ in items]; scale = 20\n"
            "    return cbs\n"
        ),
        {
            "start": 2,
            "end": 2,
            "kind": "comprehension",
            "source_code": "[lambda x: x * scale for _ in items]",
        },
        True,
    ),
    (
        "unit_starting_on_comment_line_before_first_body_stmt",
        (
            "def outer(items):\n"
            "    # leading comment before first body statement\n"
            "    for x in items:\n"
            "        cb = lambda: scale\n"
            "    scale = 10\n"
            "    return cb\n"
        ),
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_code": (
                "# leading comment before first body statement\n"
                "for x in items:\n"
                "    cb = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "neg_unit_in_multiline_class_base_header_not_treated_as_class_body",
        (
            "def outer(items, make_base):\n"
            "    scale = 1\n"
            "    class Box(\n"
            "        make_base([lambda x, item=item: x * scale + item for item in items]),\n"
            "    ):\n"
            "        nonlocal scale\n"
            "        v = Box\n"
            "    return Box\n"
        ),
        {
            "start": 4,
            "end": 4,
            "start_col": 18,
            "end_col": 75,
            "kind": "comprehension",
            "source_code": "[lambda x, item=item: x * scale + item for item in items]",
        },
        False,
    ),
    (
        "unit_in_multiline_class_base_header_captures_class_name_bound_after_header",
        (
            "def outer(items, make_base):\n"
            "    class Box(\n"
            "        make_base([lambda x, item=item: (Box, x, item) for item in items]),\n"
            "    ):\n"
            "        pass\n"
            "    return Box\n"
        ),
        {
            "start": 3,
            "end": 3,
            "start_col": 18,
            "end_col": 73,
            "kind": "comprehension",
            "source_code": "[lambda x, item=item: (Box, x, item) for item in items]",
        },
        True,
    ),
    (
        "comprehension_walrus_mutates_sibling_captured_cell",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    cb = lambda: acc\n"
            "    res = [(acc := acc + x) for x in items]\n"
            "    return res, cb\n"
        ),
        {
            "start": 4,
            "end": 4,
            "start_col": 10,
            "end_col": 43,
            "kind": "comprehension",
            "source_code": "[(acc := acc + x) for x in items]",
        },
        True,
    ),
    (
        "comprehension_walrus_mutates_enclosing_nonlocal_cell",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    def inner():\n"
            "        nonlocal acc\n"
            "        return [(acc := acc + x) for x in items]\n"
            "    return inner, acc\n"
        ),
        {
            "start": 5,
            "end": 5,
            "start_col": 15,
            "end_col": 48,
            "kind": "comprehension",
            "source_code": "[(acc := acc + x) for x in items]",
        },
        True,
    ),
    (
        "neg_local_class_staticmethod_first_param_not_aliased_as_self",
        (
            "def outer(log):\n"
            "    total = 0\n"
            "    class Worker:\n"
            "        def peek(self):\n"
            "            return total\n"
            "        @staticmethod\n"
            "        def run(items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                total += x * 2\n"
            "                log(x)\n"
            "    return Worker\n"
        ),
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x * 2\n"
                "    log(x)\n"
            ),
        },
        False,
    ),
    (
        "if_branch_with_suppress_body_store_not_definite",
        (
            "def outer(cond, suppress, d, items):\n"
            "    if cond:\n"
            "        with suppress(KeyError):\n"
            "            scale = d['x']\n"
            "    else:\n"
            "        scale = 2\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v, x=x: v * scale + x)\n"
            "    return cbs\n"
        ),
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda v, x=x: v * scale + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_single_branch_if_del_and_return_preserves_fallthrough_binding",
        (
            "def outer(cond, items):\n"
            "    scale = 2\n"
            "    if cond:\n"
            "        del scale\n"
            "        return []\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v, x=x: v * scale + x)\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda v, x=x: v * scale + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_try_finally_without_handlers_definitely_binds_on_fallthrough",
        (
            "def outer(cleanup, items):\n"
            "    try:\n"
            "        scale = 2\n"
            "    finally:\n"
            "        cleanup()\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v, x=x: v * scale + x)\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda v, x=x: v * scale + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_same_line_identifier_boundary_in_unit_cols_fallback",
        (
            "def outer(items):\n"
            "    prefix_cbs = [lambda x: x * scale for _ in items]; scale = 10; cbs = [lambda x: x * scale for _ in items]\n"
            "    return cbs\n"
        ),
        {
            "start": 2,
            "end": 2,
            "kind": "basic_block",
            "source_code": "cbs = [lambda x: x * scale for _ in items]",
        },
        False,
    ),
    (
        "pos_complex_expr_walrus_mutating_outer_local_read_after_unit",
        (
            "def outer(a, b):\n"
            "    acc = 0\n"
            "    res = (acc := a + b) * 2\n"
            "    return res + acc\n"
        ),
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 28,
            "kind": "complex_expr",
            "source_code": "(acc := a + b) * 2",
        },
        True,
    ),
    (
        "neg_complex_expr_capturing_stable_outer_local",
        (
            "def outer(a, b):\n"
            "    scale = 10\n"
            "    fn_pair = (lambda x: x * scale, a + b)\n"
            "    return fn_pair\n"
        ),
        {
            "start": 3,
            "end": 3,
            "start_col": 14,
            "end_col": 42,
            "kind": "complex_expr",
            "source_code": "(lambda x: x * scale, a + b)",
        },
        False,
    ),
]


@pytest.mark.parametrize(
    "scenario_id,src,unit_spec,expect_hazard",
    _CLOSURE_CELL_SCENARIOS,
    ids=[s[0] for s in _CLOSURE_CELL_SCENARIOS],
)
def test_severed_closure_cell_scenarios(
    scenario_id: str,
    src: str,
    unit_spec: dict[str, object],
    expect_hazard: bool,
) -> None:
    """Verifies positive/negative severed_closure_cell scenarios and lenient->strict monotonicity."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope

    _ = scenario_id
    compile(src, f"<{scenario_id}>", "exec")
    parsed_tree = ast.parse(src)
    unit = dict(unit_spec)
    unit["source_text"] = src
    scope = inspect_single_unit_scope(unit, tree=parsed_tree)
    if expect_hazard:
        assert "severed_closure_cell" in scope["control_flow_hazards"]
        assert scope["is_control_flow_safe"] is False
        reasons = scope.get("closure_hazard_reasons", [])
        assert reasons, f"Expected specific closure_hazard_reasons for {scenario_id}"
        assert "unresolved_span_bounds" not in reasons, (
            f"Scenario {scenario_id} triggered unresolved_span_bounds instead of a semantic check"
        )
        assert "invalid_lexical_declaration" not in reasons
    else:
        assert "severed_closure_cell" not in scope["control_flow_hazards"]
        assert not scope.get("closure_hazard_reasons", [])

    unit_lenient = dict(unit, skip_pre_unit_closures=True)
    scope_lenient = inspect_single_unit_scope(unit_lenient, tree=parsed_tree)
    if "severed_closure_cell" in scope_lenient["control_flow_hazards"]:
        assert "severed_closure_cell" in scope["control_flow_hazards"]


def test_closure_cell_binding_and_refactoring_integration(tmp_path: Path) -> None:
    """Verifies end-to-end refactoring, skip reasons, and nonlocal promotion."""
    from pydoppelgangerhunt.fixer import (
        generate_refactoring_patch,
        inspect_single_unit_scope,
        resolve_clone_pair_outputs,
        synthesize_shared_helper_code,
    )

    # 1. Generator subroutine inside nested closure with outer nonlocal declaration
    gen_nl_src = (
        "def outer_a(items):\n"
        "    total = 0\n"
        "    def gen_a():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total += x * 2\n"
        "            yield x\n"
        "    return gen_a, lambda: total\n"
        "\n"
        "def outer_b(items):\n"
        "    total = 10\n"
        "    def gen_b():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total += x * 2\n"
        "            yield x\n"
        "    return gen_b, lambda: total\n"
    )
    f_gen_nl = tmp_path / "gen_nl.py"
    f_gen_nl.write_text(gen_nl_src, encoding="utf-8")
    u_gen1 = {
        "file": str(f_gen_nl),
        "start": 5,
        "end": 7,
        "name": "gen_a:For",
        "kind": "compound_block",
    }
    u_gen2 = {
        "file": str(f_gen_nl),
        "start": 14,
        "end": 16,
        "name": "gen_b:For",
        "kind": "compound_block",
    }
    assert collect_downstream_read_names(gen_nl_src, u_gen1, candidates={"total", "x"}) == {"total"}
    ds_nl_all = collect_downstream_read_names(gen_nl_src, u_gen1, candidates=None)
    assert ds_nl_all is not None and "total" in ds_nl_all
    sc_gen_nl = inspect_single_unit_scope(u_gen1, repo_root=str(tmp_path))
    assert "severed_closure_cell" in sc_gen_nl["control_flow_hazards"]
    assert (
        generate_refactoring_patch(
            [(1.0, u_gen1, u_gen2)], repo_root=str(tmp_path), replace_clones=True
        )
        == ""
    )

    # Synchronous nonlocal mutation without suspension safely refactors
    sync_nl_src = (
        "def outer_a(items):\n"
        "    total = 0\n"
        "    def worker_a():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total += x * 2\n"
        "            log(x)\n"
        "    return worker_a, lambda: total\n"
        "\n"
        "def outer_b(items):\n"
        "    total = 10\n"
        "    def worker_b():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total += x * 2\n"
        "            log(x)\n"
        "    return worker_b, lambda: total\n"
    )
    f_sync_nl = tmp_path / "sync_nl.py"
    f_sync_nl.write_text(sync_nl_src, encoding="utf-8")
    u_sync1 = {
        "file": str(f_sync_nl),
        "start": 5,
        "end": 7,
        "name": "worker_a:For",
        "kind": "compound_block",
    }
    u_sync2 = {
        "file": str(f_sync_nl),
        "start": 14,
        "end": 16,
        "name": "worker_b:For",
        "kind": "compound_block",
    }
    sc_sync1 = inspect_single_unit_scope(u_sync1, repo_root=str(tmp_path))
    assert "severed_closure_cell" not in sc_sync1["control_flow_hazards"]
    patch_sync_nl = generate_refactoring_patch(
        [(1.0, u_sync1, u_sync2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "total = _shared_" in patch_sync_nl

    # 2. Conditional store to pre-bound nonlocal promotes variable to input instead of None
    cond_nl_src = (
        "def run_a(items):\n"
        "    total = 5\n"
        "    def worker():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            if x > 0:\n"
        "                total = x * 2\n"
        "    worker()\n"
        "    return total\n"
    )
    sc_cond_nl = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_text": cond_nl_src,
            "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
        },
        tree=ast.parse(cond_nl_src),
    )
    assert "total" in sc_cond_nl["inputs"]
    assert "total" in sc_cond_nl["outputs"]
    assert "total" not in sc_cond_nl["conditional_outputs"]
    assert "severed_closure_cell" not in sc_cond_nl["control_flow_hazards"]

    # Unbound nonlocal is not promoted to input and triggers severed_closure_cell
    unbound_nl_src = (
        "def outer_unb():\n"
        "    def inner_unb(items):\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            if x > 0:\n"
        "                total = x * 2\n"
        "    total = 0\n"
        "    return inner_unb\n"
    )
    sc_unbound_nl = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": unbound_nl_src,
            "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
        },
        tree=ast.parse(unbound_nl_src),
    )
    assert "total" not in sc_unbound_nl["inputs"]
    assert "total" in sc_unbound_nl["conditional_outputs"]
    assert "severed_closure_cell" in sc_unbound_nl["control_flow_hazards"]

    # Branch-only or deleted pre-unit nonlocal (including post-def del or sibling del) is not promoted to input
    for bad_init_stmt, post_stmt in (
        ("if flag:\n        total = 5", "return inner_bd"),
        ("total = 5\n    del total", "return inner_bd"),
        ("total = 5", "del total\n    return inner_bd"),
        (
            "total = 5\n    def clear():\n        nonlocal total\n        del total",
            "return inner_bd, clear",
        ),
    ):
        branch_or_del_nl_src = (
            "def outer_bd(items, flag):\n"
            f"    {bad_init_stmt}\n"
            "    def inner_bd():\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total = x * 2\n"
            f"    {post_stmt}\n"
        )
        bd_tree = ast.parse(branch_or_del_nl_src)
        for_node = next(n for n in ast.walk(bd_tree) if isinstance(n, ast.For))
        sc_bd_nl = inspect_single_unit_scope(
            {
                "start": for_node.lineno,
                "end": for_node.end_lineno,
                "kind": "compound_block",
                "source_text": branch_or_del_nl_src,
                "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
            },
            tree=bd_tree,
        )
        assert "total" not in sc_bd_nl["inputs"]
        assert "total" in sc_bd_nl["conditional_outputs"]
        assert "severed_closure_cell" in sc_bd_nl["control_flow_hazards"]

    # Lexical shadowing (mid local unbound at inner) and global in mid do not promote outer cell
    for shadow_or_glob_src in (
        (
            "def outer_sg():\n"
            "    total = 5\n"
            "    def mid():\n"
            "        def inner(items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                if x > 0:\n"
            "                    total = x * 2\n"
            "        total = 0\n"
            "        return inner\n"
            "    return mid\n"
        ),
        (
            "def outer_sg():\n"
            "    def mid():\n"
            "        global total\n"
            "        total = 10\n"
            "        def inner(items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                if x > 0:\n"
            "                    total = x * 2\n"
            "        return inner\n"
            "    total = 5\n"
            "    return mid\n"
        ),
    ):
        sg_tree = ast.parse(shadow_or_glob_src)
        sg_for = next(n for n in ast.walk(sg_tree) if isinstance(n, ast.For))
        sc_sg = inspect_single_unit_scope(
            {
                "start": sg_for.lineno,
                "end": sg_for.end_lineno,
                "kind": "compound_block",
                "source_text": shadow_or_glob_src,
                "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
            },
            tree=sg_tree,
        )
        assert "total" not in sc_sg["inputs"]
        assert "severed_closure_cell" in sc_sg["control_flow_hazards"]

    # Post-span del inside enclosing loop unbinds nonlocal for subsequent iterations
    loop_post_del_src = (
        "def outer_loop_del():\n"
        "    total = 5\n"
        "    def inner(groups):\n"
        "        nonlocal total\n"
        "        for items in groups:\n"
        "            for x in items:\n"
        "                if x > 0:\n"
        "                    total = x * 2\n"
        "            del total\n"
        "    return inner\n"
    )
    sc_lpd = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_text": loop_post_del_src,
            "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
        },
        tree=ast.parse(loop_post_del_src),
    )
    assert "total" not in sc_lpd["inputs"]
    assert "severed_closure_cell" in sc_lpd["control_flow_hazards"]

    # Conditional live-in read/AugAssign followed by unconditional write or branch-only pre_bound
    for cond_live_nl_src, u_s, u_e, u_code in (
        (
            "def outer_late_nl():\n"
            "    def inner(items):\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total += x * 2\n"
            "        total = 0\n"
            "    total = 10\n"
            "    return inner\n",
            4,
            7,
            "for x in items:\n    if x > 0:\n        total += x * 2\ntotal = 0\n",
        ),
        (
            "def outer_branch_aug(items, flag):\n"
            "    if flag:\n"
            "        total = 5\n"
            "    def inner():\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total += x * 2\n"
            "    return inner\n",
            6,
            8,
            "for x in items:\n    if x > 0:\n        total += x * 2\n",
        ),
    ):
        sc_cl = inspect_single_unit_scope(
            {
                "start": u_s,
                "end": u_e,
                "kind": "compound_block",
                "source_text": cond_live_nl_src,
                "source_code": u_code,
            },
            tree=ast.parse(cond_live_nl_src),
        )
        assert "total" not in sc_cl["inputs"]
        assert "severed_closure_cell" in sc_cl["control_flow_hazards"]

    # synthesize_shared_helper_code(closure_strictness='lenient') does not mutate caller unit dicts
    lenient_u1 = dict(u_sync1)
    lenient_u2 = dict(u_sync2)
    _ = synthesize_shared_helper_code(
        lenient_u1,
        lenient_u2,
        repo_root=str(tmp_path),
        closure_strictness="lenient",
    )
    assert "skip_pre_unit_closures" not in lenient_u1
    assert "skip_pre_unit_closures" not in lenient_u2

    # Precomputed scopes respect changed closure_strictness and skip_pre_unit_closures
    pre_clos_gen_src = (
        "def gen_pre_a(items):\n"
        "    step = 1\n"
        "    cb = lambda: step\n"
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
        "    return cb\n"
        "\n"
        "def gen_pre_b(items):\n"
        "    step = 2\n"
        "    cb = lambda: step\n"
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
        "    return cb\n"
    )
    f_pre_clos = tmp_path / "pre_clos_gen.py"
    f_pre_clos.write_text(pre_clos_gen_src, encoding="utf-8")
    u_pre_g1 = {
        "file": str(f_pre_clos),
        "start": 4,
        "end": 6,
        "name": "gen_pre_a:For",
        "kind": "compound_block",
        "source_code": "for x in items:\n    step = x + 1\n    yield step\n",
    }
    u_pre_g2 = {
        "file": str(f_pre_clos),
        "start": 12,
        "end": 14,
        "name": "gen_pre_b:For",
        "kind": "compound_block",
        "source_code": "for x in items:\n    step = x + 1\n    yield step\n",
    }
    strict_sc1 = inspect_single_unit_scope(u_pre_g1, repo_root=str(tmp_path))
    strict_sc2 = inspect_single_unit_scope(u_pre_g2, repo_root=str(tmp_path))
    strict_combined = analyze_unit_variable_scope(
        u_pre_g1, u_pre_g2, repo_root=str(tmp_path), scope1=strict_sc1, scope2=strict_sc2
    )
    assert strict_sc1["skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in strict_sc1["control_flow_hazards"]

    # Lenient synthesis and output resolution with strict precomputed scopes matches uncached lenient
    synth_lenient_no_pre = synthesize_shared_helper_code(
        u_pre_g1, u_pre_g2, repo_root=str(tmp_path), closure_strictness="lenient"
    )
    synth_lenient_with_pre = synthesize_shared_helper_code(
        u_pre_g1,
        u_pre_g2,
        repo_root=str(tmp_path),
        scope1=strict_sc1,
        scope2=strict_sc2,
        scope=strict_combined,
        closure_strictness="lenient",
    )
    assert synth_lenient_no_pre != ""
    assert synth_lenient_with_pre == synth_lenient_no_pre

    lenient_sc1 = inspect_single_unit_scope(
        {**u_pre_g1, "skip_pre_unit_closures": True}, repo_root=str(tmp_path)
    )
    lenient_sc2 = inspect_single_unit_scope(
        {**u_pre_g2, "skip_pre_unit_closures": True}, repo_root=str(tmp_path)
    )
    assert lenient_sc1["skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in lenient_sc1["control_flow_hazards"]

    res_lenient_matching_pre = resolve_clone_pair_outputs(
        u_pre_g1,
        u_pre_g2,
        lenient_sc1,
        lenient_sc2,
        repo_root=str(tmp_path),
        skip_pre_unit_closures=True,
        return_details=True,
    )
    res_lenient_with_strict_pre = resolve_clone_pair_outputs(
        u_pre_g1,
        u_pre_g2,
        strict_sc1,
        strict_sc2,
        repo_root=str(tmp_path),
        skip_pre_unit_closures=True,
        return_details=True,
    )
    assert res_lenient_matching_pre.outputs is not None
    assert res_lenient_with_strict_pre == res_lenient_matching_pre

    # Strict resolution and synthesis with lenient precomputed scopes matches uncached strict
    strict_combined_from_lenient = analyze_unit_variable_scope(
        u_pre_g1,
        u_pre_g2,
        repo_root=str(tmp_path),
        scope1=lenient_sc1,
        scope2=lenient_sc2,
    )
    assert strict_combined_from_lenient["skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in strict_combined_from_lenient["control_flow_hazards"]
    res_strict_with_lenient_pre = resolve_clone_pair_outputs(
        u_pre_g1,
        u_pre_g2,
        lenient_sc1,
        lenient_sc2,
        repo_root=str(tmp_path),
        skip_pre_unit_closures=False,
        return_details=True,
    )
    assert res_strict_with_lenient_pre.outputs is None
    assert res_strict_with_lenient_pre.skip_reason == "severed closure cell"
    assert (
        synthesize_shared_helper_code(
            u_pre_g1,
            u_pre_g2,
            repo_root=str(tmp_path),
            scope1=lenient_sc1,
            scope2=lenient_sc2,
            closure_strictness="strict",
        )
        == ""
    )
    assert (
        generate_refactoring_patch(
            [
                (
                    1.0,
                    {**u_pre_g1, "skip_pre_unit_closures": True},
                    {**u_pre_g2, "skip_pre_unit_closures": True},
                )
            ],
            repo_root=str(tmp_path),
            replace_clones=True,
            closure_strictness="strict",
        )
        == ""
    )

    # ClassDef immediate scope with nonlocal declaration in collect_downstream_read_names
    cls_nl_src = (
        "def outer_cls_nl(items):\n"
        "    total = 0\n"
        "    class Box:\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total = x * 2\n"
        "    return Box, total\n"
    )
    assert collect_downstream_read_names(
        cls_nl_src,
        {"start": 5, "end": 6},
        candidates={"total", "x"},
    ) == {"total"}

    # 3. Late-bound closure cell end-to-end skip reason and helper synthesis block
    late_cell_src = (
        "def build_a(items):\n"
        "    callbacks = []\n"
        "    for item in items:\n"
        "        def cb(x, item=item):\n"
        "            return x * scale + item\n"
        "        callbacks.append(cb)\n"
        "    scale = 10\n"
        "    return [f(2) for f in callbacks]\n"
        "\n"
        "def build_b(items):\n"
        "    callbacks = []\n"
        "    for item in items:\n"
        "        def cb(x, item=item):\n"
        "            return x * scale + item\n"
        "        callbacks.append(cb)\n"
        "    scale = 20\n"
        "    return [f(3) for f in callbacks]\n"
    )
    f_late = tmp_path / "late_cell.py"
    f_late.write_text(late_cell_src, encoding="utf-8")
    u_late1 = {
        "file": str(f_late),
        "start": 3,
        "end": 6,
        "name": "build_a:For",
        "kind": "compound_block",
    }
    u_late2 = {
        "file": str(f_late),
        "start": 12,
        "end": 15,
        "name": "build_b:For",
        "kind": "compound_block",
    }
    sc_late1 = inspect_single_unit_scope(u_late1, repo_root=str(tmp_path))
    sc_late2 = inspect_single_unit_scope(u_late2, repo_root=str(tmp_path))
    assert "severed_closure_cell" in sc_late1["control_flow_hazards"]
    assert synthesize_shared_helper_code(u_late1, u_late2, repo_root=str(tmp_path)) == ""
    res_late = resolve_clone_pair_outputs(
        u_late1, u_late2, sc_late1, sc_late2, repo_root=str(tmp_path), return_details=True
    )
    assert res_late.outputs is None
    assert res_late.skip_reason == "severed closure cell"

    # 8. Enclosing with cleanup vs severed closure cell skip reasons
    global_with_src = (
        "def run_glob(ctx, items):\n"
        "    global shared_total\n"
        "    with ctx:\n"
        "        for x in items:\n"
        "            shared_total += x\n"
    )
    u_glob_with = {
        "start": 4,
        "end": 5,
        "kind": "compound_block",
        "source_text": global_with_src,
        "source_code": "for x in items:\n    shared_total += x\n",
    }
    sc_glob = inspect_single_unit_scope(u_glob_with, tree=ast.parse(global_with_src))
    assert "severed_closure_cell" in sc_glob["control_flow_hazards"]
    res_glob = resolve_clone_pair_outputs(
        u_glob_with,
        u_glob_with,
        sc_glob,
        sc_glob,
        source_text1=global_with_src,
        source_text2=global_with_src,
        return_details=True,
    )
    assert res_glob.outputs is None
    assert res_glob.skip_reason == "severed closure cell"

    nl_with_src = (
        "def outer_with(ctx, items):\n"
        "    shared_total = 0\n"
        "    def run_nl():\n"
        "        nonlocal shared_total\n"
        "        with ctx:\n"
        "            for x in items:\n"
        "                shared_total += x\n"
        "    return run_nl\n"
    )
    u_nl_with = {
        "start": 6,
        "end": 7,
        "kind": "compound_block",
        "source_text": nl_with_src,
        "source_code": "for x in items:\n    shared_total += x\n",
    }
    sc_nl_with = inspect_single_unit_scope(u_nl_with, tree=ast.parse(nl_with_src))
    assert "severed_closure_cell" not in sc_nl_with["control_flow_hazards"]
    res_nl_with = resolve_clone_pair_outputs(
        u_nl_with,
        u_nl_with,
        sc_nl_with,
        sc_nl_with,
        source_text1=nl_with_src,
        source_text2=nl_with_src,
        return_details=True,
    )
    assert res_nl_with.outputs is None
    assert res_nl_with.skip_reason == "enclosing try or with cleanup hazard"

    # Multi-level generator with severed closure cell prioritizes severed closure cell reason
    multilevel_cell_src = (
        "def outer_chain(items):\n"
        "    scale = 1\n"
        "    def inner_chain():\n"
        "        cbs = []\n"
        "        for item in items:\n"
        "            cbs.append(lambda x, item=item: x * scale + item)\n"
        "            yield item\n"
        "    scale = 20\n"
        "    return inner_chain\n"
    )
    u_ml = {
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": multilevel_cell_src,
        "source_code": (
            "for item in items:\n"
            "    cbs.append(lambda x, item=item: x * scale + item)\n"
            "    yield item\n"
        ),
    }
    sc_ml = inspect_single_unit_scope(u_ml, tree=ast.parse(multilevel_cell_src))
    assert "severed_closure_cell" in sc_ml["control_flow_hazards"]
    res_ml_gen = resolve_clone_pair_outputs(
        u_ml,
        u_ml,
        sc_ml,
        sc_ml,
        source_text1=multilevel_cell_src,
        source_text2=multilevel_cell_src,
        return_details=True,
    )
    assert res_ml_gen.outputs is None
    assert res_ml_gen.skip_reason == "severed closure cell"


def test_df_cell_hazards_numbered_checks_unit() -> None:
    """Directly tests each numbered check function (1&2, 3, 4) in _df_cell_hazards."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_unit_nonlocal_mutators,
        _extract_closure_nonlocal_mutations,
        _extract_expr_scope_free_reads,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _check_captured_cell_mutations,
        _check_intra_unit_nonlocal_mutators,
        _check_sibling_closure_sharing,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )

    # Check 1 & 2: _check_captured_cell_mutations
    src_c12 = (
        "def outer(items, param):\n"
        "    pre = 1\n"
        "    for x in items:\n"
        "        f = lambda: x + pre + late + param\n"
        "    late = 2\n"
        "    return f\n"
    )
    tree_c12 = ast.parse(src_c12)
    chain_c12 = _find_enclosing_function_chain(tree_c12, 4, 4)
    assert len(chain_c12) == 1
    info_c12 = _analyze_enclosing_fn_cells(
        chain_c12[0][2], (4, 4, None, None), unit_mutations={"f"}
    )
    # Capturing only pre and param is safe
    assert (
        _check_captured_cell_mutations(
            {"pre", "param"}, {"pre", "param", "items"}, [info_c12]
        )
        is False
    )
    # Capturing loop-stored 'x' triggers Check 1
    assert (
        _check_captured_cell_mutations({"x"}, {"items"}, [info_c12])
        is True
    )
    # Capturing 'late' (bound after span) triggers Check 2
    assert (
        _check_captured_cell_mutations({"late"}, {"late", "items"}, [info_c12])
        is True
    )

    # Check 3: _check_intra_unit_nonlocal_mutators
    src_c3 = (
        "def outer():\n"
        "    count = 0\n"
        "    def inc():\n"
        "        nonlocal count\n"
        "        count += 1\n"
        "    inc()\n"
        "    return count\n"
    )
    tree_c3 = ast.parse(src_c3)
    chain_c3 = _find_enclosing_function_chain(tree_c3, 2, 5)
    info_c3 = _analyze_enclosing_fn_cells(chain_c3[0][2], (2, 5, None, None))
    unit_stmts_c3 = ast.parse("count = 0\ndef inc():\n    nonlocal count\n    count += 1\n").body
    mutators_c3 = _collect_unit_nonlocal_mutators(unit_stmts_c3, mod_tree=tree_c3)
    assert (
        _check_intra_unit_nonlocal_mutators(
            unit_stmts_c3,
            mutators_c3,
            {"count", "inc"},
            set(),
            [info_c3],
            is_subroutine=True,
        )
        is True
    )
    # When count is local to unit and not read after unit, Check 3 is False
    src_c3_safe = (
        "def outer():\n"
        "    count = 0\n"
        "    def inc():\n"
        "        nonlocal count\n"
        "        count += 1\n"
        "    return 42\n"
    )
    tree_c3_safe = ast.parse(src_c3_safe)
    chain_c3_safe = _find_enclosing_function_chain(tree_c3_safe, 2, 5)
    info_c3_safe = _analyze_enclosing_fn_cells(chain_c3_safe[0][2], (2, 5, None, None))
    assert (
        _check_intra_unit_nonlocal_mutators(
            unit_stmts_c3,
            mutators_c3,
            {"count", "inc"},
            set(),
            [info_c3_safe],
            is_subroutine=True,
        )
        is False
    )

    # Check 4: _check_sibling_closure_sharing
    src_c4 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def get_acc():\n"
        "        return acc\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(get_acc())\n"
        "    return acc, out\n"
    )
    tree_c4 = ast.parse(src_c4)
    chain_c4 = _find_enclosing_function_chain(tree_c4, 6, 8)
    info_c4 = _analyze_enclosing_fn_cells(chain_c4[0][2], (6, 8, None, None))
    assert (
        _check_sibling_closure_sharing(
            {"items", "acc", "out", "get_acc"},
            {"acc", "x"},
            [info_c4],
        )
        is True
    )
    # When get_acc is not an input to the unit, Check 4 is False
    assert (
        _check_sibling_closure_sharing(
            {"items", "acc", "out"},
            {"acc", "x"},
            [info_c4],
        )
        is False
    )

    # _extract_closure_nonlocal_mutations scans nested def headers (e.g. walrus in default)
    fn_with_nested_hdr = ast.parse(
        "def make():\n"
        "    nonlocal acc\n"
        "    def inner(v=(acc := acc + 1)):\n"
        "        return v\n"
        "    return inner\n"
    ).body[0]
    assert isinstance(fn_with_nested_hdr, ast.FunctionDef)
    assert "acc" in _extract_closure_nonlocal_mutations(fn_with_nested_hdr)

    # _extract_expr_scope_free_reads excludes first comprehension iterable when include_header=False
    gen_expr = ast.parse("(x * scale for x in seq)").body[0].value  # type: ignore[attr-defined]
    assert _extract_expr_scope_free_reads(gen_expr, include_header=False) == {"scale"}
    assert _extract_expr_scope_free_reads(gen_expr, include_header=True) == {"scale", "seq"}

    # Ephemeral unit modules in _collect_unit_nonlocal_mutators do not pollute caches
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
        _scope_parent_maps,
        clear_dataflow_cache,
    )

    clear_dataflow_cache()
    _collect_unit_nonlocal_mutators(ast.parse("x = 1\n").body, mod_tree=None)
    assert not any(
        isinstance(k, tuple) and any(isinstance(el, ast.Module) for el in k)
        for k in _scope_cell_info_cache
    )
    assert not any(isinstance(k, ast.Module) for k in _scope_parent_maps)

    # _evict_tree_scopes removes tuple keys when an AST node sits at an interior index (e.g. index 3)
    subtree_fn = ast.parse("def sub_fn():\n    return 1\n").body[0]
    sentinel_key: tuple[object, ...] = ("prefix", 1, frozenset({"x"}), subtree_fn, "suffix")
    _scope_cell_info_cache[sentinel_key] = "cached_val"
    _evict_tree_scopes(subtree_fn)
    assert sentinel_key not in _scope_cell_info_cache

    # _find_enclosing_function_chain excludes inner functions at greater col_offset on u_start
    same_line_tree = ast.parse(
        "def outer(cond):\n"
        "    if cond:\n"
        "        def inner():\n"
        "            return 1\n"
    )
    chain_same_line = _find_enclosing_function_chain(
        same_line_tree, 3, 4, u_start_col=4
    )
    assert [getattr(fn, "name", None) for _, _, fn in chain_same_line] == ["outer"]

    # _find_enclosing_function_chain excludes decorated inner function when unit is inside decorator
    dec_tree = ast.parse(
        "def outer(items):\n"
        "    @dec([x * 2 for x in items])\n"
        "    def inner():\n"
        "        return 1\n"
    )
    chain_dec = _find_enclosing_function_chain(dec_tree, 2, 2)
    assert [getattr(fn, "name", None) for _, _, fn in chain_dec] == ["outer"]

    # _find_enclosing_function_chain excludes inner function when unit lies inside its multi-line header
    hdr_tree = ast.parse(
        "def outer(items):\n"
        "    scale = 1\n"
        "    def inner(\n"
        "        default_cbs=[lambda x, item=item: x * scale + item for item in items],\n"
        "    ):\n"
        "        scale = 2\n"
        "        return default_cbs\n"
    )
    chain_hdr = _find_enclosing_function_chain(hdr_tree, 4, 4)
    assert [getattr(fn, "name", None) for _, _, fn in chain_hdr] == ["outer"]

    # _node_earliest_start_pos and multiline decorator span_in_hdr tracking
    from pydoppelgangerhunt.fixer._df_closures import _node_earliest_start_pos

    mdec_tree = ast.parse(
        "def outer(items, reg):\n"
        "    @reg(\n"
        "        [x for x in items],\n"
        "    )\n"
        "    def inner():\n"
        "        return post_var\n"
    )
    outer_fn = mdec_tree.body[0]
    inner_fn = outer_fn.body[0]  # type: ignore[attr-defined]
    assert _node_earliest_start_pos(inner_fn) == (2, 4)
    info_mdec = _analyze_enclosing_fn_cells(outer_fn, (3, 3, 8, 26))
    assert "inner" not in info_mdec.pre_bound
    assert "inner" in info_mdec.post_mutated
    assert "post_var" in info_mdec.post_reads

    # Decorated def as first statement of enclosing function starts at @ column
    first_dec_tree = ast.parse(
        "def o():\n"
        "    @d\n"
        "    def i():\n"
        "        return 1\n"
        "    return i\n"
    )
    chain_first_dec = _find_enclosing_function_chain(
        first_dec_tree, 2, 4, u_start_col=4
    )
    assert [getattr(fn, "name", None) for _, _, fn in chain_first_dec] == ["o"]

    # PEP 695 type_params on inner function are included in info.fn_locals
    import sys  # pylint: disable=import-outside-toplevel

    if sys.version_info >= (3, 12):
        tp_src = (
            "def outer(items):\n"
            "    T = 1\n"
            "    def inner[T]():\n"
            "        for x in items:\n"
            "            f = lambda: T\n"
            "        return f\n"
            "    return inner, T\n"
        )
        tp_tree = ast.parse(tp_src)
        tp_chain = _find_enclosing_function_chain(tp_tree, 4, 5)
        tp_inner_info = _analyze_enclosing_fn_cells(
            tp_chain[-1][2], (4, 5, None, None)
        )
        assert "T" in tp_inner_info.fn_locals


def test_expr_traversal_subscript_compare_unary_fstring_and_ifexp() -> None:
    """Verifies _extract_expr_closure_and_load_info and hazard checks across nested expressions."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_expr_closure_and_load_info,
    )

    expr_cases = [
        ("callbacks[index_cb()]", {"callbacks"}, {"index_cb"}),
        ("items[index]", {"items", "index"}, set()),
        ("left_cb < right_cb()", {"left_cb"}, {"right_cb"}),
        ("not unary_cb(arg_cb)", {"arg_cb"}, {"unary_cb"}),
        ("f'{fmt_cb(val_cb):{spec_cb}}'", {"val_cb", "spec_cb"}, {"fmt_cb"}),
        ("body_cb if cond_cb() else alt_cb", {"body_cb", "alt_cb"}, {"cond_cb"}),
    ]
    for expr_src, expected_loads, expected_calls in expr_cases:
        expr_node = ast.parse(expr_src, mode="eval").body
        called: set[str] = set()
        _, _, loaded = _extract_expr_closure_and_load_info(
            [expr_node], called_names_out=called
        )
        assert expected_loads <= loaded, expr_src
        assert expected_calls <= called, expr_src

    # Closure references and calls nested inside Subscript slice, Compare, UnaryOp, and JoinedStr
    nested_hazard_cases = [
        (
            "def outer(items, callbacks):\n"
            "    idx = 0\n"
            "    def get_idx():\n"
            "        return idx\n"
            "    def pick():\n"
            "        return callbacks[get_idx()]\n"
            "    out = []\n"
            "    for x in items:\n"
            "        idx += x\n"
            "        out.append(pick())\n"
            "    return out\n",
            8,
            10,
            "for x in items:\n    idx += x\n    out.append(pick())\n",
        ),
        (
            "def outer(items):\n"
            "    limit = 0\n"
            "    def get_limit():\n"
            "        return limit\n"
            "    def check(v):\n"
            "        return v < get_limit()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        limit += x\n"
            "        out.append(check(x))\n"
            "    return out\n",
            8,
            10,
            "for x in items:\n    limit += x\n    out.append(check(x))\n",
        ),
        (
            "def outer(items):\n"
            "    flag = False\n"
            "    def get_flag():\n"
            "        return flag\n"
            "    def inverted():\n"
            "        return not get_flag()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        flag = bool(x)\n"
            "        out.append(inverted())\n"
            "    return out\n",
            8,
            10,
            "for x in items:\n    flag = bool(x)\n    out.append(inverted())\n",
        ),
        (
            "def outer(items):\n"
            "    tag = 'a'\n"
            "    def get_tag():\n"
            "        return tag\n"
            "    def render():\n"
            "        return f'[{get_tag()}]'\n"
            "    out = []\n"
            "    for x in items:\n"
            "        tag = str(x)\n"
            "        out.append(render())\n"
            "    return out\n",
            8,
            10,
            "for x in items:\n    tag = str(x)\n    out.append(render())\n",
        ),
    ]
    for src, s_line, e_line, u_code in nested_hazard_cases:
        unit = {
            "start": s_line,
            "end": e_line,
            "kind": "compound_block",
            "source_text": src,
            "source_code": u_code,
        }
        scope = inspect_single_unit_scope(unit, tree=ast.parse(src))
        assert "severed_closure_cell" in scope["control_flow_hazards"]


def test_differential_cell_mutations_vs_ordinary_bindings(tmp_path: Path) -> None:
    """Differential execution tests distinguishing cell mutations, object mutation, locals, branches, and deletion."""
    # 1a. Rebinding a variable captured by a nested function (post/loop rebinding blocked)
    rebind_hazard_a = (
        "def entry(items):\n"
        "    scale = 1\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda v: v + scale)\n"
        "        z = x * 2\n"
        "    scale = 10\n"
        "    return [cb(5) for cb in cbs], z\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "rebind_captured_blocked",
        rebind_hazard_a,
        (4, 6),
        expect_patch=False,
        call_args_list=[([1, 2, 3],)],
    )

    # 1b. Pre-bound unchanging variable captured by nested function (refactors and matches)
    rebind_safe_a = (
        "def entry(items):\n"
        "    scale = 10\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        step = x * scale\n"
        "        cbs.append(lambda v, s=step: v + scale + s)\n"
        "    return [cb(1) for cb in cbs]\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "prebound_captured_safe",
        rebind_safe_a,
        (4, 6),
        expect_patch=True,
        call_args_list=[([1, 2, 3],), ([],)],
    )

    # 2. Mutating an object referenced by a captured variable without rebinding the variable
    obj_mut_a = (
        "def entry(items):\n"
        "    history = []\n"
        "    snap = lambda: list(history)\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        history.append(x * 2)\n"
        "        total += x\n"
        "    return total, snap(), history\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "object_mutation_without_rebinding",
        obj_mut_a,
        (5, 7),
        expect_patch=True,
        call_args_list=[([2, 4, 6],), ([],)],
    )

    # 3. Rebinding a name local to a nested function rather than to the enclosing function
    inner_local_a = (
        "def entry(items):\n"
        "    scale = 5\n"
        "    def helper_local(v):\n"
        "        scale = v * 10\n"
        "        return scale + 1\n"
        "    out = []\n"
        "    for x in items:\n"
        "        cb = lambda y, x=x: y + scale + x\n"
        "        out.append(cb(2))\n"
        "    return out, helper_local(3), scale\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "inner_function_local_rebinding",
        inner_local_a,
        (7, 9),
        expect_patch=True,
        call_args_list=[([1, 2],)],
    )

    # 4. Conditional assignment inside try/except that may raise before completing
    cond_try_a = (
        "def entry(should_fail):\n"
        "    def risky():\n"
        "        if should_fail:\n"
        "            raise ValueError('boom')\n"
        "        return 99\n"
        "    value = 1\n"
        "    try:\n"
        "        value = risky()\n"
        "    except Exception:\n"
        "        pass\n"
        "    getter = lambda: value\n"
        "    return getter(), value\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "conditional_try_preserve_prebound",
        cond_try_a,
        (7, 10),
        expect_patch=True,
        call_args_list=[(True,), (False,)],
    )

    # 5. Deleting a binding, including exception-handler targets cleared by Python
    exc_clear_a = (
        "def entry(items):\n"
        "    err = 'initial'\n"
        "    try:\n"
        "        raise RuntimeError('fail')\n"
        "    except RuntimeError as err:\n"
        "        pass\n"
        "    err = 'recovered'\n"
        "    out = []\n"
        "    for x in items:\n"
        "        cb = lambda v, x=x: f'{err}:{v + x}'\n"
        "        out.append(cb(1))\n"
        "    return out\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "except_handler_target_rebound",
        exc_clear_a,
        (9, 11),
        expect_patch=True,
        call_args_list=[([10, 20],)],
    )


def test_nested_scope_reads_do_not_falsely_escape_mutator() -> None:
    """Verifies unexecuted or synchronous intra-unit references to a nonlocal mutator do not count as escaping."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_shallow_loads_and_calls,
    )

    # 1. Unused nested function calling increment()
    unused_caller_src = (
        "def outer(items):\n"
        "    if items:\n"
        "        count = 0\n"
        "        def increment():\n"
        "            nonlocal count\n"
        "            count += 1\n"
        "        def unused():\n"
        "            increment()\n"
        "    return len(items)\n"
    )
    u_unused = {
        "start": 2,
        "end": 8,
        "kind": "compound_block",
        "source_text": unused_caller_src,
        "source_code": (
            "if items:\n"
            "    count = 0\n"
            "    def increment():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    def unused():\n"
            "        increment()\n"
        ),
    }
    sc_unused = inspect_single_unit_scope(
        u_unused, tree=ast.parse(unused_caller_src)
    )
    assert "severed_closure_cell" not in sc_unused["control_flow_hazards"]

    # 2. Synchronous nested caller invoked inside unit, where neither count nor increment escapes
    sync_caller_src = (
        "def outer(items):\n"
        "    if items:\n"
        "        count = 0\n"
        "        def increment():\n"
        "            nonlocal count\n"
        "            count += 1\n"
        "        def run_once():\n"
        "            increment()\n"
        "        run_once()\n"
        "    return len(items)\n"
    )
    u_sync = {
        "start": 2,
        "end": 9,
        "kind": "compound_block",
        "source_text": sync_caller_src,
        "source_code": (
            "if items:\n"
            "    count = 0\n"
            "    def increment():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    def run_once():\n"
            "        increment()\n"
            "    run_once()\n"
        ),
    }
    sc_sync = inspect_single_unit_scope(u_sync, tree=ast.parse(sync_caller_src))
    assert "severed_closure_cell" not in sc_sync["control_flow_hazards"]

    # 3. Escaping nested caller (run_once returned/used after unit while count is also read) -> hazard
    escaped_caller_src = (
        "def outer(items):\n"
        "    if items:\n"
        "        count = 0\n"
        "        def increment():\n"
        "            nonlocal count\n"
        "            count += 1\n"
        "        def run_once():\n"
        "            increment()\n"
        "    return run_once, count\n"
    )
    u_escaped = {
        "start": 2,
        "end": 8,
        "kind": "compound_block",
        "source_text": escaped_caller_src,
        "source_code": (
            "if items:\n"
            "    count = 0\n"
            "    def increment():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    def run_once():\n"
            "        increment()\n"
        ),
    }
    sc_escaped = inspect_single_unit_scope(
        u_escaped, tree=ast.parse(escaped_caller_src)
    )
    assert "severed_closure_cell" in sc_escaped["control_flow_hazards"]

    # 4. _extract_shallow_loads_and_calls stops at nested def body but inspects default/decorator headers
    stmts = ast.parse(
        "def inner(cb=header_ref):\n"
        "    body_ref()\n"
        "    return body_val\n"
    ).body
    non_call, _ = _extract_shallow_loads_and_calls(stmts)
    assert "header_ref" in non_call
    assert "body_ref" not in non_call
    assert "body_val" not in non_call


def test_partial_column_inference_all_four_combinations_and_utf8() -> None:
    """Verifies _unit_cols and _resolve_unit_col_bounds across all 4 (start_col, end_col) combinations and UTF-8."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_span_index import (
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer.scope import _unit_cols

    # 1. Multiple statements on a single ASCII line
    ascii_line = "    x = 1; y = x + 2; z = y * 3\n"
    target_snippet = "y = x + 2"
    exp_s = len("    x = 1; ".encode("utf-8"))
    exp_e = len("    x = 1; y = x + 2".encode("utf-8"))

    for s_in, e_in in [
        (None, None),
        (None, exp_e),
        (exp_s, None),
        (exp_s, exp_e),
    ]:
        unit_dict: dict[str, Any] = {"source_code": target_snippet}
        if s_in is not None:
            unit_dict["start_col"] = s_in
        if e_in is not None:
            unit_dict["end_col"] = e_in
        got_s, got_e = _unit_cols(unit_dict, [ascii_line])
        assert (got_s, got_e) == (exp_s, exp_e), (s_in, e_in)

    # 2. _resolve_unit_col_bounds with identical AST statements on the same line
    dup_src = "def f(a):\n    a = a + 1; a = a + 1\n    return a\n"
    dup_tree = ast.parse(dup_src)
    cand_stmt = ast.parse("a = a + 1\n").body
    # First statement is cols 4..13, second statement is cols 15..24; ambiguous when both cols are None
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, None, None) == (
        None,
        None,
    )
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, 4, None) == (
        4,
        13,
    )
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, 15, None) == (
        15,
        24,
    )
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, None, 24) == (
        15,
        24,
    )
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, 4, 13) == (
        4,
        13,
    )

    # 3. Non-ASCII UTF-8 multi-statement line across all 4 coordinate combinations
    utf8_src = (
        "def calc():\n"
        "    café = '☕'; résumé = café + '!'; return lambda: résumé\n"
    )
    utf8_line = "    café = '☕'; résumé = café + '!'; return lambda: résumé\n"
    utf8_snippet = "résumé = café + '!'"
    pos = utf8_line.index(utf8_snippet)
    u8_s = len(utf8_line[:pos].encode("utf-8"))
    u8_e = len(utf8_line[: pos + len(utf8_snippet)].encode("utf-8"))

    for s_in, e_in in [
        (None, None),
        (None, u8_e),
        (u8_s, None),
        (u8_s, u8_e),
    ]:
        u_utf8: dict[str, Any] = {
            "start": 2,
            "end": 2,
            "kind": "sliding_window",
            "source_text": utf8_src,
            "source_code": utf8_snippet,
        }
        if s_in is not None:
            u_utf8["start_col"] = s_in
        if e_in is not None:
            u_utf8["end_col"] = e_in
        sc = inspect_single_unit_scope(u_utf8, tree=ast.parse(utf8_src))
        assert sc["inputs"] == ["café"]
        assert sc["outputs"] == ["résumé"]
