"""Clone matching algorithms, SourcererCC bound pruning, and target scanner."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union, overload

from pydoppelgangerhunt.config import (
    canonical_path_key,
    find_python_files,
    normalize_path_string,
    paths_match_boundary,
)
from pydoppelgangerhunt.baseline import _safe_index_frequency, _safe_total_units
from pydoppelgangerhunt.parser import harvest_file_units

DEFAULT_STOP_SHINGLES: Set[Tuple[str, ...]] = {
    # Main guard boilerplate: if __name__ == "__main__":
    ("Module", "If", "Compare"),
    ("If", "Compare", "Pass"),
    ("If", "Compare", "Expr"),
    ("Compare", "Pass", "VAR"),
    ("Compare", "Expr", "VAR"),
    ("Pass", "VAR", "Eq"),
    ("VAR", "Eq", "CONST_str"),
    ("Eq", "CONST_str", "Load"),
    ("Eq", "CONST_str", "Call"),
    # Standard logger & logging calls:
    ("Module", "Expr", "Call"),
    ("Expr", "Call", "ATTR"),
    ("Call", "ATTR", "CONST_str"),
    ("Call", "ATTR", "VAR"),
    ("ATTR", "CONST_str", "VAR"),
    ("ATTR", "VAR", "VAR"),
    ("ATTR", "VAR", "Load"),
    ("Call", "Load", "ATTR"),
    ("Load", "Call", "ATTR"),
    ("Load", "ATTR", "VAR"),
    # Standard try/except error handling & logging / pass:
    ("Module", "Try", "Pass"),
    ("Module", "Try", "Expr"),
    ("Try", "Pass", "ExceptHandler"),
    ("Try", "Expr", "ExceptHandler"),
    ("Pass", "ExceptHandler", "VAR"),
    ("ExceptHandler", "VAR", "Pass"),
    ("ExceptHandler", "VAR", "Expr"),
    ("ExceptHandler", "Call", "VAR"),
    ("Expr", "ExceptHandler", "Call"),
    # Exception raising boilerplate:
    ("Module", "Raise", "Call"),
    ("Raise", "Call", "VAR"),
}


def get_boilerplate_stop_shingles() -> Set[Tuple[str, ...]]:
    """Returns a copy of the default set of boilerplate stop-shingles."""
    return set(DEFAULT_STOP_SHINGLES)


def _normalize_matcher_file(file_str: Optional[str]) -> str:
    """Normalizes unit file path string for robust matching, preserving cell anchors."""
    return canonical_path_key(file_str, strip_anchor=False)


def _split_exemption_endpoint(ep: str) -> Tuple[str, Optional[str]]:
    """Splits an exemption endpoint into (file_path, symbol_name), accounting for Windows drive letters."""
    if ":" not in ep:
        return ep, None
    f_part, sym_part = ep.rsplit(":", 1)
    if "/" in sym_part or "\\" in sym_part or (len(f_part) == 1 and f_part.isalpha()):
        return ep, None
    return f_part, sym_part


def _normalize_exemption_endpoint(
    ep: str, repo_root: Optional[Union[str, Path]] = None
) -> str:
    """Normalizes an exemption endpoint by normalizing the file portion relative to repo root."""
    f_part, sym_part = _split_exemption_endpoint(ep)
    if repo_root is not None:
        try:
            p = Path(f_part)
            if p.is_absolute():
                f_part = str(p.resolve().relative_to(Path(repo_root).resolve()))
        except (ValueError, OSError):
            pass
    norm_file = _normalize_matcher_file(f_part)
    if sym_part is not None:
        return f"{norm_file}:{sym_part}"
    return norm_file


def _sorted_pair(a: str, b: str) -> Tuple[str, str]:
    """Returns an order-invariant sorted 2-tuple of two string keys."""
    return (a, b) if a <= b else (b, a)




def lcs_alignment_similarity(
    tokens_a: Sequence[str],
    tokens_b: Sequence[str],
    threshold: Optional[float] = None,
) -> float:
    """Computes CCAligner-style Longest Common Subsequence alignment similarity between two token sequences."""
    if not tokens_a or not tokens_b:
        return 0.0
    len_a = len(tokens_a)
    len_b = len(tokens_b)
    # SourcererCC theoretical upper-bound check: max possible similarity
    max_possible = (2.0 * min(len_a, len_b)) / (len_a + len_b)
    if threshold is not None and max_possible < threshold:
        return 0.0

    if len_a < len_b:
        tokens_a, tokens_b = tokens_b, tokens_a
        len_a, len_b = len_b, len_a
    m = len_b
    dp = [0] * (m + 1)
    for tok_a in tokens_a:
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if tok_a == tokens_b[j - 1]:
                dp[j] = prev + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev = temp
    return (2.0 * dp[m]) / (len_a + len_b)


def call_sequence_similarity(seq1: Sequence[str], seq2: Sequence[str]) -> float:
    """Computes sequence alignment similarity between two function/method call traces."""
    if not seq1 or not seq2:
        return 0.0
    return lcs_alignment_similarity(seq1, seq2)


def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Computes Jaccard similarity score between two shingle sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def tfidf_multiset_jaccard_similarity(
    vec_a: Dict[Any, int], vec_b: Dict[Any, int], idf_weights: Dict[Any, float]
) -> float:
    """Computes TF-IDF weighted multiset Jaccard similarity between two token frequency vectors."""
    if not vec_a or not vec_b:
        return 0.0
    all_keys = set(vec_a.keys()).union(vec_b.keys())
    intersection_sum = sum(
        min(vec_a.get(k, 0), vec_b.get(k, 0)) * idf_weights.get(k, 1.0) for k in all_keys
    )
    union_sum = sum(
        max(vec_a.get(k, 0), vec_b.get(k, 0)) * idf_weights.get(k, 1.0) for k in all_keys
    )
    return intersection_sum / union_sum if union_sum > 0 else 0.0


def multiset_jaccard_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
    """Computes multiset Jaccard (Ruzicka) similarity between two token frequency vectors."""
    return tfidf_multiset_jaccard_similarity(vec_a, vec_b, idf_weights={})


def tfidf_jaccard_similarity(
    set_a: Set[Any], set_b: Set[Any], idf_weights: Dict[Any, float]
) -> float:
    """Computes TF-IDF weighted Jaccard similarity between two shingle sets."""
    if not idf_weights:
        return jaccard_similarity(set_a, set_b)
    return tfidf_multiset_jaccard_similarity(
        dict.fromkeys(set_a, 1), dict.fromkeys(set_b, 1), idf_weights
    )


def compute_pair_similarity(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    *,
    bag_of_tokens: bool = False,
    tfidf: bool = False,
    gapped_tolerance: bool = False,
    call_sequences: bool = False,
    idf_weights: Optional[Dict[Any, float]] = None,
    threshold: Optional[float] = None,
) -> float:
    """Computes similarity between two code units under configured metric."""
    if call_sequences:
        c1 = u1.get("calls", [])
        c2 = u2.get("calls", [])
        if len(c1) >= 3 and len(c2) >= 3:
            return call_sequence_similarity(c1, c2)
        return 0.0

    len1 = u1.get("token_count", len(u1.get("tokens", [])))
    len2 = u2.get("token_count", len(u2.get("tokens", [])))
    if threshold is not None and len1 > 0 and len2 > 0:
        max_possible = (2.0 * min(len1, len2)) / (len1 + len2)
        if max_possible < threshold:
            return 0.0

    if tfidf and idf_weights is not None:
        if bag_of_tokens:
            return tfidf_multiset_jaccard_similarity(
                u1.get("vector", {}), u2.get("vector", {}), idf_weights
            )
        return tfidf_jaccard_similarity(
            u1.get("shingles", set()), u2.get("shingles", set()), idf_weights
        )
    if gapped_tolerance:
        # SourcererCC Theorem: LCS similarity >= T requires Multiset Jaccard >= T / (2 - T)
        if threshold is not None and threshold > 0:
            min_m_jaccard = threshold / (2.0 - threshold)
            m_jaccard = multiset_jaccard_similarity(u1.get("vector", {}), u2.get("vector", {}))
            if m_jaccard < min_m_jaccard:
                return 0.0
        return lcs_alignment_similarity(
            u1.get("tokens", []), u2.get("tokens", []), threshold=threshold
        )
    if bag_of_tokens:
        return multiset_jaccard_similarity(u1.get("vector", {}), u2.get("vector", {}))
    return jaccard_similarity(u1.get("shingles", set()), u2.get("shingles", set()))


def _update_merged_unit_columns(
    target_unit: Dict[str, Any],
    donor_unit: Dict[str, Any],
    target_start: int,
    target_end: int,
    donor_start: int,
    donor_end: int,
) -> None:
    """Propagates outer column boundaries when merging two adjacent or overlapping units."""
    for key, d_pos, t_pos, is_expanded, agg in (
        ("start_col", donor_start, target_start, donor_start < target_start, min),
        ("end_col", donor_end, target_end, donor_end > target_end, max),
    ):
        if is_expanded:
            target_unit[key] = donor_unit.get(key)
        elif d_pos == t_pos and donor_unit.get(key) is not None:
            curr_val = target_unit.get(key)
            donor_val = donor_unit[key]
            target_unit[key] = agg(curr_val, donor_val) if curr_val is not None else donor_val


def _check_column_bounds_relationship(
    p_start: int,
    p_end: int,
    c_start: int,
    c_end: int,
    p_unit: Dict[str, Any],
    c_unit: Dict[str, Any],
) -> Tuple[bool, bool]:
    """Evaluates whether child unit columns are enclosed within and/or strictly smaller than parent.

    Returns:
        A tuple of (is_enclosed, is_strictly_smaller).
    """
    p_sc, c_sc = p_unit.get("start_col"), c_unit.get("start_col")
    p_ec, c_ec = p_unit.get("end_col"), c_unit.get("end_col")

    if c_start == p_start and p_sc is not None and c_sc is not None:
        if int(c_sc) < int(p_sc):
            return False, False

    if c_end == p_end and p_ec is not None and c_ec is not None:
        if int(c_ec) > int(p_ec):
            return False, False

    strictly_smaller = (
        (c_start == p_start and p_sc is not None and c_sc is not None and int(c_sc) > int(p_sc))
        or (c_end == p_end and p_ec is not None and c_ec is not None and int(c_ec) < int(p_ec))
    )
    return True, strictly_smaller


def merge_adjacent_clones(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    line_tolerance: int = 2,
    threshold: Optional[float] = None,
    bag_of_tokens: bool = False,
    tfidf: bool = False,
    idf_weights: Optional[Dict[Any, float]] = None,
    gapped_tolerance: bool = False,
) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """Merges adjacent and overlapping clone pairs into maximal continuous clone regions."""
    if not clones:
        return []

    canonical: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for sim, u1, u2 in clones:
        f1 = _normalize_matcher_file(u1.get("file"))
        f2 = _normalize_matcher_file(u2.get("file"))
        k1 = (f1, int(u1.get("start") or 1), str(u1.get("name") or ""))
        k2 = (f2, int(u2.get("start") or 1), str(u2.get("name") or ""))
        if k1 <= k2:
            canonical.append((sim, dict(u1), dict(u2)))
        else:
            canonical.append((sim, dict(u2), dict(u1)))

    merged_clones = canonical
    changed = True

    while changed:
        changed = False
        new_clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        skip: Set[int] = set()

        for i, (sim_a, u1_a, u2_a) in enumerate(merged_clones):
            if i in skip:
                continue

            current_u1 = dict(u1_a)
            current_u2 = dict(u2_a)
            merged_with_any = False

            for j in range(i + 1, len(merged_clones)):
                if j in skip:
                    continue
                _sim_b, u1_b, u2_b = merged_clones[j]

                f1_curr = _normalize_matcher_file(current_u1.get("file"))
                f1_b = _normalize_matcher_file(u1_b.get("file"))
                f2_curr = _normalize_matcher_file(current_u2.get("file"))
                f2_b = _normalize_matcher_file(u2_b.get("file"))
                if f1_curr != f1_b or f2_curr != f2_b:
                    continue

                fn1_a = str(current_u1.get("name") or "").split(":", maxsplit=1)[0]
                fn1_b = str(u1_b.get("name") or "").split(":", maxsplit=1)[0]
                fn2_a = str(current_u2.get("name") or "").split(":", maxsplit=1)[0]
                fn2_b = str(u2_b.get("name") or "").split(":", maxsplit=1)[0]
                if fn1_a != fn1_b or fn2_a != fn2_b:
                    continue

                u1_b_start = int(u1_b.get("start") or 1)
                u1_b_end = int(u1_b.get("end") or u1_b_start)
                curr_u1_start = int(current_u1.get("start") or 1)
                curr_u1_end = int(current_u1.get("end") or curr_u1_start)

                u2_b_start = int(u2_b.get("start") or 1)
                u2_b_end = int(u2_b.get("end") or u2_b_start)
                curr_u2_start = int(current_u2.get("start") or 1)
                curr_u2_end = int(current_u2.get("end") or curr_u2_start)

                adj1 = (u1_b_start <= curr_u1_end + line_tolerance) and (
                    u1_b_end >= curr_u1_start - line_tolerance
                )
                adj2 = (u2_b_start <= curr_u2_end + line_tolerance) and (
                    u2_b_end >= curr_u2_start - line_tolerance
                )

                if not (adj1 and adj2):
                    continue

                _update_merged_unit_columns(
                    current_u1, u1_b, curr_u1_start, curr_u1_end, u1_b_start, u1_b_end
                )
                current_u1["start"] = min(curr_u1_start, u1_b_start)
                current_u1["end"] = max(curr_u1_end, u1_b_end)
                current_u1["lines"] = current_u1["end"] - current_u1["start"] + 1
                current_u1["shingles"] = set(current_u1.get("shingles") or set()).union(
                    u1_b.get("shingles") or set()
                )
                current_u1["token_count"] = max(
                    int(current_u1.get("token_count") or 0),
                    int(u1_b.get("token_count") or 0),
                )
                current_u1["tokens"] = list(current_u1.get("tokens") or []) + list(
                    u1_b.get("tokens") or []
                )
                current_u1["name"] = f"{fn1_a}:merged_{current_u1['start']}-{current_u1['end']}"
                v1_a = current_u1.get("vector", {})
                v1_b = u1_b.get("vector", {})
                current_u1["vector"] = {k: v1_a.get(k, 0) + v1_b.get(k, 0) for k in set(v1_a).union(v1_b)}

                _update_merged_unit_columns(
                    current_u2, u2_b, curr_u2_start, curr_u2_end, u2_b_start, u2_b_end
                )
                current_u2["start"] = min(curr_u2_start, u2_b_start)
                current_u2["end"] = max(curr_u2_end, u2_b_end)
                current_u2["lines"] = current_u2["end"] - current_u2["start"] + 1
                current_u2["shingles"] = set(current_u2.get("shingles") or set()).union(
                    u2_b.get("shingles") or set()
                )
                current_u2["token_count"] = max(
                    int(current_u2.get("token_count") or 0),
                    int(u2_b.get("token_count") or 0),
                )
                current_u2["tokens"] = list(current_u2.get("tokens") or []) + list(
                    u2_b.get("tokens") or []
                )
                current_u2["name"] = f"{fn2_a}:merged_{current_u2['start']}-{current_u2['end']}"
                v2_a = current_u2.get("vector", {})
                v2_b = u2_b.get("vector", {})
                current_u2["vector"] = {k: v2_a.get(k, 0) + v2_b.get(k, 0) for k in set(v2_a).union(v2_b)}

                skip.add(j)
                changed = True
                merged_with_any = True

            combined_sim = compute_pair_similarity(
                current_u1,
                current_u2,
                bag_of_tokens=bag_of_tokens,
                tfidf=tfidf,
                gapped_tolerance=gapped_tolerance,
                idf_weights=idf_weights,
            )
            if not merged_with_any:
                combined_sim = sim_a
            new_clones.append((combined_sim, current_u1, current_u2))

        merged_clones = new_clones

    if threshold is not None:
        merged_clones = [c for c in merged_clones if c[0] >= threshold]

    merged_clones.sort(key=lambda x: x[0], reverse=True)
    return merged_clones


def suppress_subclones(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    sim_tolerance: float = 0.05,
) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """Eliminates redundant sub-clones geometrically contained within an enclosing parent clone."""
    if len(clones) <= 1:
        return clones

    suppressed_indices: Set[int] = set()

    for i, (sim_parent, p1, p2) in enumerate(clones):
        if i in suppressed_indices:
            continue
        f1_p = _normalize_matcher_file(p1.get("file"))
        f2_p = _normalize_matcher_file(p2.get("file"))

        for j, (sim_child, c1, c2) in enumerate(clones):
            if i == j or j in suppressed_indices:
                continue
            f1_c = _normalize_matcher_file(c1.get("file"))
            f2_c = _normalize_matcher_file(c2.get("file"))

            direct_match = (f1_p == f1_c and f2_p == f2_c)
            reverse_match = (f1_p == f2_c and f2_p == f1_c)

            if not (direct_match or reverse_match):
                continue

            c1_corr = c1 if direct_match else c2
            c2_corr = c2 if direct_match else c1

            p1_start = int(p1.get("start") or 1)
            p1_end = int(p1.get("end") or p1_start)
            p2_start = int(p2.get("start") or 1)
            p2_end = int(p2.get("end") or p2_start)
            c1_start = int(c1_corr.get("start") or 1)
            c1_end = int(c1_corr.get("end") or c1_start)
            c2_start = int(c2_corr.get("start") or 1)
            c2_end = int(c2_corr.get("end") or c2_start)

            c1_enclosed = p1_start <= c1_start and c1_end <= p1_end
            c2_enclosed = p2_start <= c2_start and c2_end <= p2_end

            if not (c1_enclosed and c2_enclosed):
                continue

            enc1, small1 = _check_column_bounds_relationship(
                p1_start, p1_end, c1_start, c1_end, p1, c1_corr
            )
            if not enc1:
                continue

            enc2, small2 = _check_column_bounds_relationship(
                p2_start, p2_end, c2_start, c2_end, p2, c2_corr
            )
            if not enc2:
                continue

            is_strictly_smaller = (
                (c1_start > p1_start or c1_end < p1_end or small1)
                or (c2_start > p2_start or c2_end < p2_end or small2)
            )
            if not is_strictly_smaller:
                continue

            if sim_parent >= sim_child - sim_tolerance:
                suppressed_indices.add(j)

    return [c for idx, c in enumerate(clones) if idx not in suppressed_indices]




def _lookup_calib_freq(calib_freqs: Optional[Dict[Any, Any]], key: Any) -> Optional[Any]:
    """Looks up a shingle key in calibration frequencies supporting tuples, tagged keys, and legacy keys."""
    if not calib_freqs or not isinstance(calib_freqs, dict):
        return None
    val = calib_freqs.get(key)
    if val is not None:
        return val
    if isinstance(key, (tuple, list)):
        try:
            dumped = json.dumps(list(key))
            val = calib_freqs.get("t:" + dumped)
            if val is not None:
                return val
            val = calib_freqs.get(dumped)
            if val is not None:
                return val
        except (TypeError, ValueError):
            pass
    elif isinstance(key, str):
        val = calib_freqs.get("s:" + key)
        if val is not None:
            return val
    elif isinstance(key, bool):
        val = calib_freqs.get("b:" + ("1" if key else "0"))
        if val is not None:
            return val
    elif isinstance(key, int):
        val = calib_freqs.get("i:" + str(key))
        if val is not None:
            return val
    elif isinstance(key, float):
        val = calib_freqs.get("f:" + str(key))
        if val is not None:
            return val
    return None


def _compute_max_posting_len(
    total_units: int,
    freq: Optional[float],
    min_corpus: int,
    fallback: Optional[int] = None,
) -> Optional[int]:
    """Calculates max posting length for frequency-based pruning, handling overflow safely."""
    if freq is not None and total_units >= min_corpus:
        try:
            posting_float = total_units * freq
            if math.isfinite(posting_float):
                return max(2, int(math.ceil(posting_float)))
        except (ValueError, TypeError, OverflowError):
            pass
    return fallback


def _add_candidate_pairs(
    candidate_pairs: Set[Tuple[int, int]],
    indices: Sequence[int],
    diff_unit_indices: Optional[Set[int]] = None,
) -> None:
    """Populates candidate pairs from posting list, filtering by diff units if active."""
    if diff_unit_indices is None:
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i + 1:]:
                candidate_pairs.add((min(idx1, idx2), max(idx1, idx2)))
        return

    diff_in: List[int] = []
    diff_out: List[int] = []
    for idx in indices:
        if idx in diff_unit_indices:
            diff_in.append(idx)
        else:
            diff_out.append(idx)

    if not diff_in:
        return

    for i, idx1 in enumerate(diff_in):
        for idx2 in diff_in[i + 1:]:
            candidate_pairs.add((min(idx1, idx2), max(idx1, idx2)))

    for idx1 in diff_in:
        for idx2 in diff_out:
            candidate_pairs.add((min(idx1, idx2), max(idx1, idx2)))


def _build_calibration_metadata(
    units: Sequence[Dict[str, Any]],
    *,
    max_index_frequency: Optional[float],
    min_corpus_size: Optional[int],
    filter_stop_shingles: bool,
    stop_shingles: Optional[Set[Any]],
    bag_of_tokens: bool,
    call_sequences: bool,
) -> Dict[str, Any]:
    """Helper to compute baseline corpus calibration dictionary."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration
    return compute_corpus_calibration(
        units,
        max_index_frequency=max_index_frequency,
        min_corpus_size=min_corpus_size,
        filter_stop_shingles=filter_stop_shingles,
        stop_shingles=stop_shingles,
        bag_of_tokens=bag_of_tokens,
        call_sequences=call_sequences,
    )


def _worker_harvest_file(task_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Worker task wrapper for process pool executor."""
    return harvest_file_units(**task_kwargs)


@overload
def scan_target(
    target_dir: str,
    *,
    return_calibration: Literal[False] = False,
    diff_files: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]: ...


@overload
def scan_target(
    target_dir: str,
    *,
    return_calibration: Literal[True],
    diff_files: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], Dict[str, Any]]: ...


@overload
def scan_target(
    target_dir: str,
    *,
    return_calibration: bool = False,
    diff_files: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Union[
    List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], Dict[str, Any]],
]: ...


def scan_target(
    target_dir: str,
    *,
    min_lines: int = 8,
    min_tokens: int = 15,
    threshold: float = 0.90,
    repo_root: Optional[Union[str, Path]] = None,
    diff_files: Optional[Sequence[str]] = None,
    excludes: Optional[List[str]] = None,
    functions_only: bool = False,
    sliding_window: bool = False,
    window_size: int = 5,
    blind_indexing: bool = False,
    merge_subtrees: bool = False,
    complex_expressions: bool = False,
    min_expr_complexity: int = 4,
    clause_level: bool = False,
    data_tables: bool = False,
    strip_annotations: bool = True,
    nms: bool = False,
    class_level: bool = False,
    blind_literals: bool = False,
    bag_of_tokens: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    tfidf: bool = False,
    harvest_closures: bool = False,
    commutative: bool = False,
    comprehensions: bool = False,
    idioms: bool = False,
    abstract_expressions: bool = False,
    gapped_tolerance: bool = False,
    call_sequences: bool = False,
    audit_tests: bool = False,
    exemptions: Optional[List[Tuple[str, str]]] = None,
    workers: Optional[int] = None,
    sort_by: str = "similarity",
    top_n: Optional[int] = None,
    include_notebooks: bool = False,
    strip_docstrings: bool = True,
    max_index_frequency: Optional[float] = 0.25,
    filter_stop_shingles: bool = False,
    stop_shingles: Optional[Set[Any]] = None,
    min_corpus_size: Optional[int] = None,
    corpus_calibration: Optional[Dict[str, Any]] = None,
    return_calibration: bool = False,
    **kwargs: Any,
) -> Union[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], Dict[str, Any]]]:
    units: List[Dict[str, Any]] = []
    if repo_root is not None:
        effective_repo_root = Path(repo_root)
    else:
        try:
            target_path = Path(target_dir).resolve()
            cwd = Path.cwd().resolve()
            try:
                target_path.relative_to(cwd)
                effective_repo_root = cwd
            except ValueError:
                effective_repo_root = target_path if target_path.is_dir() else target_path.parent
        except (ValueError, OSError):
            effective_repo_root = Path.cwd()
    file_list = find_python_files(
        target_dir,
        excludes=excludes,
        audit_tests=audit_tests,
        include_notebooks=include_notebooks,
    )

    effective_workers = workers if workers is not None else 1
    # Multi-core process pool when requested or on large repos
    if effective_workers > 1 and len(file_list) >= 10:
        task_list = [
            {
                "file_path": str(p),
                "repo_root": str(effective_repo_root),
                "min_lines": min_lines,
                "min_tokens": min_tokens,
                "functions_only": functions_only,
                "sliding_window": sliding_window,
                "window_size": window_size,
                "blind_indexing": blind_indexing,
                "complex_expressions": complex_expressions,
                "min_expr_complexity": min_expr_complexity,
                "clause_level": clause_level,
                "data_tables": data_tables,
                "strip_annotations": strip_annotations,
                "class_level": class_level,
                "blind_literals": blind_literals,
                "filter_boilerplate": filter_boilerplate,
                "consistent_renaming": consistent_renaming,
                "harvest_closures": harvest_closures,
                "commutative": commutative,
                "comprehensions": comprehensions,
                "idioms": idioms,
                "abstract_expressions": abstract_expressions,
                "strip_docstrings": strip_docstrings,
            }
            for p in file_list
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for file_units in executor.map(_worker_harvest_file, task_list):
                units.extend(file_units)
    else:
        for p in file_list:
            units.extend(
                harvest_file_units(
                    str(p),
                    str(effective_repo_root),
                    min_lines=min_lines,
                    min_tokens=min_tokens,
                    functions_only=functions_only,
                    sliding_window=sliding_window,
                    window_size=window_size,
                    blind_indexing=blind_indexing,
                    complex_expressions=complex_expressions,
                    min_expr_complexity=min_expr_complexity,
                    clause_level=clause_level,
                    data_tables=data_tables,
                    strip_annotations=strip_annotations,
                    class_level=class_level,
                    blind_literals=blind_literals,
                    filter_boilerplate=filter_boilerplate,
                    consistent_renaming=consistent_renaming,
                    harvest_closures=harvest_closures,
                    commutative=commutative,
                    comprehensions=comprehensions,
                    idioms=idioms,
                    abstract_expressions=abstract_expressions,
                    strip_docstrings=strip_docstrings,
                )
            )

    diff_unit_indices: Optional[Set[int]] = None
    if diff_files is not None:
        norm_diff_files_list = [
            normalize_path_string(d_file, strip_anchor=False)
            for d_file in diff_files
            if d_file
        ]
        norm_diff_files_set = set(norm_diff_files_list)
        diff_unit_indices = set()
        for idx, u in enumerate(units):
            u_file = normalize_path_string(str(u.get("file") or ""), strip_anchor=False)
            if not u_file:
                continue
            if u_file in norm_diff_files_set:
                diff_unit_indices.add(idx)
                continue
            for d_file in norm_diff_files_list:
                if paths_match_boundary(u_file, d_file):
                    diff_unit_indices.add(idx)
                    break
        if not diff_unit_indices:
            if return_calibration:
                return [], _build_calibration_metadata(
                    units,
                    max_index_frequency=max_index_frequency,
                    min_corpus_size=min_corpus_size,
                    filter_stop_shingles=filter_stop_shingles,
                    stop_shingles=stop_shingles,
                    bag_of_tokens=bag_of_tokens,
                    call_sequences=call_sequences,
                )
            return []

    effective_stop_shingles: Set[Any] = set()
    if filter_stop_shingles:
        effective_stop_shingles.update(DEFAULT_STOP_SHINGLES)
    if stop_shingles is not None:
        effective_stop_shingles.update(stop_shingles)
    if not isinstance(corpus_calibration, dict):
        corpus_calibration = None
    if corpus_calibration is not None:
        calib_stops = corpus_calibration.get("global_stop_shingles")
        if calib_stops and isinstance(calib_stops, (list, set, tuple)):
            from pydoppelgangerhunt.baseline import _deep_tuple
            for s in calib_stops:
                try:
                    effective_stop_shingles.add(_deep_tuple(s))
                except (TypeError, ValueError, RecursionError):
                    continue

    idf_weights: Dict[Any, float] = {}
    if tfidf and units:
        calib_units_tfidf = 0
        if corpus_calibration is not None:
            local_units_count = len(diff_unit_indices) if diff_unit_indices is not None else len(units)
            calib_units_tfidf = _safe_total_units(corpus_calibration.get("total_units"))
            corpus_size = local_units_count + calib_units_tfidf
            active_indices: Union[Set[int], range] = (
                diff_unit_indices
                if diff_unit_indices is not None
                else range(len(units))
            )
        else:
            corpus_size = len(units)
            active_indices = range(len(units))

        df_counts: Dict[Any, int] = {}
        for idx in active_indices:
            u = units[idx]
            keys = u.get("vector", {}).keys() if bag_of_tokens else u["shingles"]
            for k in keys:
                if effective_stop_shingles and k in effective_stop_shingles:
                    continue
                df_counts[k] = df_counts.get(k, 0) + 1
        calib_freqs_raw = (
            corpus_calibration.get("shingle_frequencies")
            if corpus_calibration is not None
            else None
        )
        calib_freqs: Dict[Any, Any] = (
            calib_freqs_raw if isinstance(calib_freqs_raw, dict) else {}
        )
        for k, df in df_counts.items():
            calib_df = _lookup_calib_freq(calib_freqs, k)
            try:
                parsed_calib = int(calib_df or 0)
                if calib_units_tfidf > 0 and parsed_calib > 0:
                    valid_calib_df = min(calib_units_tfidf, parsed_calib)
                else:
                    valid_calib_df = 0
            except (ValueError, TypeError, OverflowError):
                valid_calib_df = 0
            combined_df = df + valid_calib_df
            try:
                weight = math.log((1.0 + corpus_size) / (1.0 + combined_df)) + 1.0
                idf_weights[k] = max(0.0, weight)
            except (ValueError, TypeError, OverflowError):
                idf_weights[k] = 1.0

    shingle_index: Dict[Any, List[int]] = {}
    for idx, u in enumerate(units):
        if call_sequences:
            index_keys = list(set(u.get("calls", [])))
        elif bag_of_tokens:
            index_keys = list(u.get("vector", {}).keys())
        else:
            index_keys = list(u["shingles"])
        for sh in index_keys:
            if effective_stop_shingles and sh in effective_stop_shingles:
                continue
            shingle_index.setdefault(sh, []).append(idx)

    raw_calib_min_corpus = (
        corpus_calibration.get("min_corpus_size")
        if corpus_calibration is not None and "min_corpus_size" in corpus_calibration
        else min_corpus_size
    )
    effective_min_corpus = (
        raw_calib_min_corpus
        if raw_calib_min_corpus is not None
        else (4 if filter_stop_shingles else 30)
    )

    candidate_pairs: Set[Tuple[int, int]] = set()
    calib_freqs_map: Optional[Dict[Any, Any]] = None
    calib_units = 0
    if corpus_calibration is not None:
        calib_units = _safe_total_units(corpus_calibration.get("total_units"))
        local_units_count = len(diff_unit_indices) if diff_unit_indices is not None else len(units)
        total_corpus_units = local_units_count + calib_units
        calib_freqs_raw = corpus_calibration.get("shingle_frequencies")
        calib_freqs_map = (
            calib_freqs_raw if isinstance(calib_freqs_raw, dict) else None
        )
        raw_calib_max_freq = (
            corpus_calibration.get("max_index_frequency")
            if "max_index_frequency" in corpus_calibration
            else max_index_frequency
        )
        active_max_freq = _safe_index_frequency(raw_calib_max_freq)
        max_posting_len = _compute_max_posting_len(
            total_corpus_units, active_max_freq, effective_min_corpus, fallback=None
        )
    else:
        active_max_freq = _safe_index_frequency(max_index_frequency)
        max_posting_len = _compute_max_posting_len(
            len(units), active_max_freq, effective_min_corpus, fallback=len(units) + 1
        ) or (len(units) + 1)

    effective_max_posting = max_posting_len
    if (
        effective_max_posting is None
        and active_max_freq is not None
        and len(units) >= effective_min_corpus
    ):
        effective_max_posting = _compute_max_posting_len(
            len(units), active_max_freq, effective_min_corpus, fallback=None
        )

    for sh, u_indices in shingle_index.items():
        if len(u_indices) <= 1:
            continue
        if diff_unit_indices is not None:
            df_local = sum(1 for idx in u_indices if idx in diff_unit_indices)
            if df_local == 0:
                continue
        else:
            df_local = len(u_indices)

        is_global_shingle = False
        if calib_freqs_map is not None:
            raw_global = _lookup_calib_freq(calib_freqs_map, sh)
            if raw_global is not None:
                try:
                    val = int(raw_global)
                    if calib_units > 0 and val > 0:
                        is_global_shingle = True
                        df_global = min(calib_units, val)
                    else:
                        df_global = 0
                except (ValueError, TypeError, OverflowError):
                    df_global = 0
            else:
                df_global = 0
            combined_df = df_global + df_local
        else:
            combined_df = len(u_indices)

        if effective_stop_shingles and sh in effective_stop_shingles:
            continue

        if diff_unit_indices is not None and calib_freqs_map is not None and not is_global_shingle:
            df_unmodified = len(u_indices) - df_local
            if df_unmodified > 0 and effective_max_posting is not None and (
                df_unmodified > effective_max_posting or len(u_indices) > effective_max_posting
            ):
                continue
        elif effective_max_posting is not None and combined_df > effective_max_posting:
            continue

        _add_candidate_pairs(candidate_pairs, u_indices, diff_unit_indices)

    raw_exemptions = exemptions if exemptions is not None else []
    normalized_exemptions: Set[Tuple[str, str]] = set()
    basename_exemptions: Set[Tuple[str, str]] = set()
    for pair in raw_exemptions:
        if len(pair) == 2:
            k1, k2 = pair
            ep1 = _normalize_exemption_endpoint(k1, repo_root=effective_repo_root)
            ep2 = _normalize_exemption_endpoint(k2, repo_root=effective_repo_root)
            normalized_exemptions.add(_sorted_pair(ep1, ep2))
            f1_part, sym1 = _split_exemption_endpoint(ep1)
            f2_part, sym2 = _split_exemption_endpoint(ep2)
            k1_file, _ = _split_exemption_endpoint(k1)
            k2_file, _ = _split_exemption_endpoint(k2)
            has_dir1 = ("/" in k1_file or "\\" in k1_file) and not (len(k1_file) == 1 and k1_file.isalpha())
            has_dir2 = ("/" in k2_file or "\\" in k2_file) and not (len(k2_file) == 1 and k2_file.isalpha())
            if not has_dir1 and not has_dir2:
                base_f1 = os.path.basename(f1_part)
                base_f2 = os.path.basename(f2_part)
                b1 = f"{base_f1}:{sym1}" if sym1 is not None else base_f1
                b2 = f"{base_f2}:{sym2}" if sym2 is not None else base_f2
                basename_exemptions.add(_sorted_pair(b1, b2))

    target_norm = target_dir.replace("\\", "/").strip("./").rstrip("/")
    target_pfx = f"{target_norm}/" if target_norm and target_norm != "." else ""
    target_prefixes = (
        (target_pfx, "src/", "pydoppelgangerhunt/")
        if target_pfx
        else ("src/", "pydoppelgangerhunt/")
    )

    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for i, j in candidate_pairs:
        u1, u2 = units[i], units[j]

        if audit_tests and not (u1["name"].startswith("test_") and u2["name"].startswith("test_")):
            continue

        f1 = _normalize_matcher_file(u1.get("file"))
        f2 = _normalize_matcher_file(u2.get("file"))

        if f1 == f2:
            fn1 = u1["name"].split(":")[0]
            fn2 = u2["name"].split(":")[0]
            if fn1 == fn2:
                if max(u1["start"], u2["start"]) <= min(u1["end"], u2["end"]):
                    continue
            elif (u1["start"] <= u2["start"] and u1["end"] >= u2["end"]) or (u2["start"] <= u1["start"] and u2["end"] >= u1["end"]):
                continue

        # Skip if size differs significantly
        if u1["token_count"] > 2.5 * u2["token_count"] or u2["token_count"] > 2.5 * u1["token_count"]:
            continue

        t_count1 = u1["token_count"]
        t_count2 = u2["token_count"]
        if not call_sequences and t_count1 > 0 and t_count2 > 0:
            max_bound = (2.0 * min(t_count1, t_count2)) / (t_count1 + t_count2)
            if max_bound < threshold:
                continue

        f1_pkg = f1
        f2_pkg = f2
        for pfx in target_prefixes:
            if pfx and f1_pkg.startswith(pfx):
                f1_pkg = f1_pkg[len(pfx):]
            if pfx and f2_pkg.startswith(pfx):
                f2_pkg = f2_pkg[len(pfx):]

        pair_id_rel = _sorted_pair(f"{f1}:{u1['name']}", f"{f2}:{u2['name']}")
        pair_id_pkg = _sorted_pair(f"{f1_pkg}:{u1['name']}", f"{f2_pkg}:{u2['name']}")
        pair_id_file = _sorted_pair(f1, f2)
        pair_id_file_pkg = _sorted_pair(f1_pkg, f2_pkg)
        if (
            pair_id_rel in normalized_exemptions
            or pair_id_pkg in normalized_exemptions
            or pair_id_file in normalized_exemptions
            or pair_id_file_pkg in normalized_exemptions
        ):
            continue

        if basename_exemptions:
            pair_id_base = _sorted_pair(
                f"{os.path.basename(f1)}:{u1['name']}",
                f"{os.path.basename(f2)}:{u2['name']}",
            )
            pair_id_file_base = _sorted_pair(os.path.basename(f1), os.path.basename(f2))
            if pair_id_base in basename_exemptions or pair_id_file_base in basename_exemptions:
                continue

        sim = compute_pair_similarity(
            u1,
            u2,
            bag_of_tokens=bag_of_tokens,
            tfidf=tfidf,
            gapped_tolerance=gapped_tolerance,
            call_sequences=call_sequences,
            idf_weights=idf_weights if tfidf else None,
            threshold=threshold,
        )
        if sim >= threshold:
            clones.append((sim, u1, u2))

    if merge_subtrees:
        clones = merge_adjacent_clones(
            clones,
            threshold=threshold,
            bag_of_tokens=bag_of_tokens,
            tfidf=tfidf,
            idf_weights=idf_weights if tfidf else None,
            gapped_tolerance=gapped_tolerance,
        )

    if nms:
        clones = suppress_subclones(clones)

    if sort_by == "priority":
        clones.sort(key=lambda x: compute_priority_score(x[0], x[1], x[2]), reverse=True)
    elif sort_by == "sloc":
        clones.sort(
            key=lambda x: (
                max(0, int(x[1].get("end") or int(x[1].get("start") or 1)) - int(x[1].get("start") or 1) + 1)
                + max(0, int(x[2].get("end") or int(x[2].get("start") or 1)) - int(x[2].get("start") or 1) + 1)
            ),
            reverse=True,
        )
    else:
        clones.sort(key=lambda x: x[0], reverse=True)

    if top_n is not None and top_n > 0:
        clones = clones[:top_n]

    if return_calibration:
        calib_dict = _build_calibration_metadata(
            units,
            max_index_frequency=max_index_frequency,
            min_corpus_size=min_corpus_size,
            filter_stop_shingles=filter_stop_shingles,
            stop_shingles=stop_shingles,
            bag_of_tokens=bag_of_tokens,
            call_sequences=call_sequences,
        )
        return clones, calib_dict

    return clones


def compute_priority_score(
    sim: float,
    u1: Dict[str, Any],
    u2: Dict[str, Any],
) -> float:
    """Calculates refactoring priority based on similarity, line length, and cyclomatic complexity."""
    s1 = int(u1.get("start") or 1)
    e1 = int(u1.get("end") or s1)
    s2 = int(u2.get("start") or 1)
    e2 = int(u2.get("end") or s2)
    avg_sloc = (max(0, e1 - s1 + 1) + max(0, e2 - s2 + 1)) / 2.0
    max_comp = max(1, int(u1.get("complexity") or 1), int(u2.get("complexity") or 1))
    return float(round(sim * avg_sloc * max_comp, 1))
