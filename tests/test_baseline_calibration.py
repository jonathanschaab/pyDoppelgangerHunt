"""Unit tests for baseline corpus calibration, shingle wire serialization, config hashing, and reduction metrics."""

from __future__ import annotations
import json
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)
import pytest
from pydoppelgangerhunt import (
    load_baseline,
    record_baseline,
    scan_target,
)


def test_corpus_calibration_compute_and_serialization(tmp_path: Path) -> None:
    """Verifies that compute_corpus_calibration computes accurate distributions and round-trips via baseline v1.4.0."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )

    # Construct synthetic units with known shingle frequencies
    units: List[Dict[str, Any]] = [
        {"file": f"f{i}.py", "name": f"fn_{i}", "shingles": {("Common", "A"), ("Domain", "B")}}
        for i in range(10)
    ]
    # Add a ubiquitous shingle in 8/10 units (80% > 25%)
    for i in range(8):
        sh_set = units[i]["shingles"]
        assert isinstance(sh_set, set)
        sh_set.add(("Ubiquitous", "Boilerplate"))
    # Add a rare shingle in 2/10 units (20% <= 25%)
    for i in range(2):
        sh_set = units[i]["shingles"]
        assert isinstance(sh_set, set)
        sh_set.add(("Rare", "Shingle"))

    calib = compute_corpus_calibration(units, max_index_frequency=0.25, min_corpus_size=4)
    assert calib["total_units"] == 10
    assert calib["max_index_frequency"] == 0.25
    assert ("Ubiquitous", "Boilerplate") in calib["global_stop_shingles"]
    assert ("Common", "A") in calib["global_stop_shingles"]  # 10/10 = 100% > 25%
    assert ("Rare", "Shingle") not in calib["global_stop_shingles"]  # 2/10 = 20% <= 25%
    assert calib["shingle_frequencies"][("Ubiquitous", "Boilerplate")] == 8
    assert calib["shingle_frequencies"][("Rare", "Shingle")] == 2

    # Round-trip via record_baseline and load_baseline
    base_file = tmp_path / "baseline_calib.json"
    u1 = {"file": "mod.py", "name": "f1", "tokens": ["a"]}
    u2 = {"file": "mod.py", "name": "f2", "tokens": ["b"]}
    saved = record_baseline([(1.0, u1, u2)], str(base_file), "target", 0.90, corpus_calibration=calib)
    assert Path(saved).is_file()

    raw_json = json.loads(base_file.read_text(encoding="utf-8"))
    assert raw_json["version"] in ("1.4.0", "1.5.0")
    assert "corpus_calibration" in raw_json
    assert raw_json["corpus_calibration"]["total_units"] == 10

    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    loaded_calib = loaded.corpus_calibration
    assert loaded_calib["total_units"] == 10
    assert ("Ubiquitous", "Boilerplate") in loaded_calib["global_stop_shingles"]
    assert loaded_calib["shingle_frequencies"][("Ubiquitous", "Boilerplate")] == 8
    assert loaded_calib["shingle_frequencies"][("Rare", "Shingle")] == 2

def test_differential_scan_calibrated_pruning_prevents_false_negatives(tmp_path: Path) -> None:
    """Verifies that corpus calibration prevents false-negative pruning of domain shingles in small diff runs.

    In a small diff scan of 8 units, 3 units sharing a domain shingle represent 37.5% (> 25%) locally.
    Without calibration, the shingle exceeds max_posting_len = 2 and is falsely pruned, dropping clone candidates.
    With global calibration (1000 units), the combined frequency is 3/1008 (0.3%), preserving the clone candidates.
    """
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    repo.mkdir()

    # Create 8 files, 3 of which share an identical domain function
    shared_code = (
        "def domain_logic_clone(a, b, c):\n"
        "    result = 0\n"
        "    for x in range(a):\n"
        "        if x > b:\n"
        "            result += x * c\n"
        "        else:\n"
        "            result -= x\n"
        "    return result\n"
    )

    (repo / "f1.py").write_text(shared_code, encoding="utf-8")
    (repo / "f2.py").write_text(shared_code, encoding="utf-8")
    (repo / "f3.py").write_text(shared_code, encoding="utf-8")

    # 5 files with distinct un-cloned functions to total 8 units
    for i in range(4, 9):
        code = f"def distinct_func_{i}(val):\n" + "\n".join(f"    v_{j} = val + {j}" for j in range(i)) + "\n    return val\n"
        (repo / f"f{i}.py").write_text(code, encoding="utf-8")

    # Run WITHOUT calibration: with 8 units and max_index_frequency=0.25 (min_corpus_size=4),
    # max_posting_len is max(2, ceil(8 * 0.25)) = 2.
    # The domain shingles in f1, f2, f3 appear 3 times (> 2) and are all pruned as stop shingles!
    clones_uncalibrated = scan_target(
        str(repo),
        threshold=0.90,
        min_lines=6,
        max_index_frequency=0.25,
        min_corpus_size=4,
    )
    assert len(clones_uncalibrated) == 0, "Expected 0 clones due to false-negative local frequency skew"

    # Run WITH global corpus calibration (e.g. 1000 units in full repo)
    calib = {
        "total_units": 1000,
        "max_index_frequency": 0.25,
        "min_lines": 6,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }
    clones_calibrated = scan_target(
        str(repo),
        threshold=0.90,
        min_lines=6,
        max_index_frequency=0.25,
        min_corpus_size=4,
        corpus_calibration=calib,
    )
    assert len(clones_calibrated) >= 3, f"Expected at least 3 clone pairs detected with calibration, got {len(clones_calibrated)}"

def test_differential_scan_calibrated_pruning_suppresses_global_boilerplate(tmp_path: Path) -> None:
    """Verifies that calibrated global stop shingles suppress ubiquitous repository boilerplate even in small diffs."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_boiler"
    repo.mkdir()

    # Two units that share boilerplate
    boilerplate_code = (
        "def handle_service_call(req):\n"
        "    try:\n"
        "        logger.info('Processing %s', req)\n"
        "        return req.process()\n"
        "    except Exception as exc:\n"
        "        logger.error('Failed: %s', exc)\n"
        "        raise\n"
    )
    (repo / "s1.py").write_text(boilerplate_code, encoding="utf-8")
    (repo / "s2.py").write_text(boilerplate_code, encoding="utf-8")

    # 1. Uncalibrated scan on s1.py and s2.py finds 1 clone pair
    clones_uncalib = scan_target(str(repo), threshold=0.90, min_lines=5)
    assert len(clones_uncalib) == 1

    # Harvest units to collect the boilerplate shingles
    _, calib_harvested = scan_target(str(repo), threshold=0.90, min_lines=5, return_calibration=True)
    all_boilerplate_shingles = set(calib_harvested["shingle_frequencies"].keys())
    assert len(all_boilerplate_shingles) > 0

    # Calibrate with all boilerplate shingles designated as global stop shingles
    calib = {
        "total_units": 500,
        "max_index_frequency": 0.25,
        "min_lines": 5,
        "global_stop_shingles": all_boilerplate_shingles,
        "shingle_frequencies": {sh: 300 for sh in all_boilerplate_shingles},
    }

    # Verify that passing this calibration suppresses candidate pair generation and yields 0 clones
    clones_calibrated = scan_target(
        str(repo),
        threshold=0.90,
        min_lines=5,
        corpus_calibration=calib,
    )
    assert len(clones_calibrated) == 0, "Expected boilerplate clone pair to be completely suppressed by calibrated stop shingles"

def test_baseline_calibration_edge_cases_and_security(tmp_path: Path) -> None:
    """Verifies security resilience against malformed baseline JSON and edge-case serialization."""
    from pydoppelgangerhunt._baseline_calibration import (  # pylint: disable=import-outside-toplevel
        _deserialize_shingle_key,
    )
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        load_baseline,
        record_baseline,
    )

    # 1. Non-dict JSON payloads do not crash load_baseline
    for malformed_content in [b"[1, 2, 3]", b'"a string"', b"42", b"true"]:
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_bytes(malformed_content)
        fps = load_baseline(str(malformed_file))
        assert len(fps) == 0
        assert fps.corpus_calibration is None

    # 2. Corrupted fields in corpus_calibration do not crash load_baseline
    corrupted_file = tmp_path / "corrupted_calib.json"
    corrupted_file.write_text(
        json.dumps({
            "version": "1.4.0",
            "corpus_calibration": {
                "total_units": "not_an_int",
                "max_index_frequency": "not_a_float",
                "global_stop_shingles": "not_a_list",
                "shingle_frequencies": {"k": "not_an_int"},
            },
        }),
        encoding="utf-8",
    )
    loaded_corrupt = load_baseline(str(corrupted_file))
    assert loaded_corrupt.corpus_calibration is not None
    assert loaded_corrupt.corpus_calibration["total_units"] == 0
    assert loaded_corrupt.corpus_calibration["max_index_frequency"] == 0.25

    # 3. max_index_frequency=None round-trip
    calib_none_max = {
        "total_units": 100,
        "max_index_frequency": None,
        "global_stop_shingles": {("A", "B"), ("C", "D")},
        "shingle_frequencies": {("A", "B"): 5, ("C", "D"): 10},
    }
    none_max_file = tmp_path / "none_max.json"
    record_baseline([], str(none_max_file), "target", 0.90, corpus_calibration=calib_none_max)
    loaded_none = load_baseline(str(none_max_file))
    assert loaded_none.corpus_calibration is not None
    assert loaded_none.corpus_calibration["max_index_frequency"] is None

    # 4. Nested shingle structure deserialization produces hashable tuples
    nested_res = _deserialize_shingle_key("[[1, 2], [3, 4]]")
    assert isinstance(nested_res, tuple)
    assert isinstance(nested_res[0], tuple)
    assert hash(nested_res) is not None

    # 5. Deterministic sorting in serialized output
    raw_text = none_max_file.read_text(encoding="utf-8")
    loaded_json = json.loads(raw_text)
    stops = loaded_json["corpus_calibration"]["global_stop_shingles"]
    freq_keys = list(loaded_json["corpus_calibration"]["shingle_frequencies"].keys())
    assert stops == sorted(stops, key=str)
    assert freq_keys == sorted(freq_keys)

def test_scan_target_calibration_edge_cases(tmp_path: Path) -> None:
    """Verifies that scan_target correctly handles raw list stop shingles, string keys, and small corpus bounds."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "calib_edge_repo"
    repo.mkdir()

    sample_func = (
        "def duplicated_logic(x, y):\n"
        "    val = x + 1\n"
        "    acc = val * 2\n"
        "    diff = acc - y\n"
        "    total = diff + 4\n"
        "    return total\n"
    )
    (repo / "f1.py").write_text(sample_func, encoding="utf-8")
    (repo / "f2.py").write_text(sample_func, encoding="utf-8")

    # 1. Uncalibrated scan on 2 files finds 1 clone pair
    uncalib_clones = scan_target(str(repo), min_lines=5, threshold=0.90)
    assert len(uncalib_clones) == 1

    # 2. Calibrated scan on small corpus (< effective_min_corpus=30) must NOT falsely prune clones
    small_calib: Dict[str, Any] = {
        "total_units": 2,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }
    calib_clones = scan_target(str(repo), min_lines=5, threshold=0.90, corpus_calibration=small_calib)
    assert len(calib_clones) == 1, "Small corpus with calibration must not prune valid clones"

    # 3. Handling max_index_frequency=None in corpus_calibration does not raise TypeError
    none_freq_calib: Dict[str, Any] = {
        "total_units": 2,
        "max_index_frequency": None,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }
    none_clones = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        max_index_frequency=None,
        corpus_calibration=none_freq_calib,
    )
    assert len(none_clones) == 1

    # 4. Raw list-of-lists in global_stop_shingles does not raise unhashable type: 'list'
    list_stops_calib: Dict[str, Any] = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "global_stop_shingles": [["Name", "Assign"]],
        "shingle_frequencies": {'["Name", "Assign"]': 50},
    }
    list_stop_clones = scan_target(str(repo), min_lines=5, threshold=0.90, corpus_calibration=list_stops_calib)
    assert isinstance(list_stop_clones, list)

def test_prune_baseline_preserves_corpus_calibration(tmp_path: Path) -> None:
    """Verifies that prune_baseline preserves corpus_calibration metadata in the baseline file."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        load_baseline,
        prune_baseline,
        record_baseline,
    )

    base_file = tmp_path / "baseline_for_prune.json"
    calib = {
        "total_units": 42,
        "max_index_frequency": 0.20,
        "global_stop_shingles": {("Stop", "A")},
        "shingle_frequencies": {("Stop", "A"): 15},
    }
    u1 = {"file": "mod1.py", "name": "fn1", "tokens": ["x", "1"]}
    u2 = {"file": "mod2.py", "name": "fn2", "tokens": ["y", "2"]}
    record_baseline([(1.0, u1, u2)], str(base_file), "target", 0.90, corpus_calibration=calib)

    # Prune with active clones retaining the entry
    prune_res = prune_baseline(str(base_file), [(1.0, u1, u2)], unstaged_modified_ranges={})
    assert prune_res.retained_count == 1

    # Load baseline and verify corpus_calibration is intact
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["total_units"] == 42
    assert loaded.corpus_calibration["max_index_frequency"] == 0.20
    assert ("Stop", "A") in loaded.corpus_calibration["global_stop_shingles"]
    assert loaded.corpus_calibration["shingle_frequencies"][("Stop", "A")] == 15

def test_prune_baseline_synchronizes_calibration_scope_and_hash(tmp_path: Path) -> None:
    """Verifies that prune_baseline synchronizes corpus_calibration scope and recomputes config_hash."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_calibration_config_hash,
        load_baseline,
        prune_baseline,
    )
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    f1 = src / "a.py"
    f2 = src / "b.py"
    f1.write_text("def run():\n    pass\n", encoding="utf-8")
    f2.write_text("def run():\n    pass\n", encoding="utf-8")

    u1 = {"file": "a.py", "name": "run"}
    u2 = {"file": "b.py", "name": "run"}
    base_file = tmp_path / "pruned_calib.json"

    # Baseline file where corpus_calibration initially lacked scope
    initial_calib: Dict[str, Any] = {
        "total_units": 10,
        "max_index_frequency": 0.25,
        "scope": None,
        "target_repo_relative": None,
    }
    initial_hash = compute_calibration_config_hash(initial_calib)
    initial_calib["config_hash"] = initial_hash

    raw_data = {
        "version": "1.4.0",
        "target": str(src),
        "target_repo_relative": "src",
        "path_basis": "target_relative",
        "clone_count": 1,
        "fingerprints": [
            {
                "file_a": "a.py",
                "file_b": "b.py",
                "name_a": "run",
                "name_b": "run",
                "fingerprint": "fp1",
                "structural_fingerprint": "sfp1",
            }
        ],
        "corpus_calibration": initial_calib,
        "config_hash": initial_hash,
    }
    base_file.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")

    # Prune with active clones
    prune_res = prune_baseline(
        str(base_file),
        [(1.0, u1, u2)],
        unstaged_modified_ranges={},
        target=str(src),
        repo_root=str(repo),
    )
    assert prune_res.retained_count == 1

    # Check persisted file on disk directly
    saved_data = json.loads(base_file.read_text(encoding="utf-8"))
    saved_calib = saved_data["corpus_calibration"]
    assert saved_calib["scope"] == "src"
    assert saved_calib["target_repo_relative"] == "src"
    expected_hash = compute_calibration_config_hash(saved_calib)
    assert saved_calib["config_hash"] == expected_hash
    assert saved_data["config_hash"] == expected_hash
    assert saved_calib["config_hash"] != initial_hash

    # Check loaded baseline
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["scope"] == "src"
    assert _is_calibration_mode_compatible(loaded.corpus_calibration, target_scope="src")

def test_record_baseline_calibration_hardening(tmp_path: Path) -> None:
    """Test record_baseline robustly sanitizes non-finite floats, overflow, and invalid counts."""
    from pydoppelgangerhunt.baseline import load_baseline, record_baseline  # pylint: disable=import-outside-toplevel

    u1 = {"file": "mod1.py", "name": "fn1", "structural_hash": "hash1"}
    u2 = {"file": "mod2.py", "name": "fn2", "structural_hash": "hash2"}
    clones = [(0.95, u1, u2)]

    # 1. Non-finite max_index_frequency (NaN, Inf) and negative/invalid counts
    calib_malformed = {
        "total_units": "invalid_int",
        "max_index_frequency": float("nan"),
        "global_stop_shingles": ["stop_a", ("nested", "shingle"), 12345],
        "shingle_frequencies": {
            "valid_key": 42,
            "zero_key": 0,
            "negative_key": -5,
            "invalid_count": "not_an_int",
        },
    }

    out_file = tmp_path / "baseline_hardened.json"
    record_baseline(clones, str(out_file), "target_repo", 0.90, corpus_calibration=calib_malformed)

    raw_json = json.loads(out_file.read_text(encoding="utf-8"))
    calib_data = raw_json.get("corpus_calibration")
    assert calib_data is not None
    assert calib_data["total_units"] == 0
    assert calib_data["max_index_frequency"] is None
    assert calib_data["shingle_frequencies"] == {"valid_key": 42}
    assert "stop_a" in calib_data["global_stop_shingles"]

    # 2. Inf and out-of-range floats
    calib_inf = {
        "total_units": 50,
        "max_index_frequency": float("inf"),
        "shingle_frequencies": {"k": 10},
    }
    out_inf = tmp_path / "baseline_inf.json"
    record_baseline(clones, str(out_inf), "target_repo", 0.90, corpus_calibration=calib_inf)
    data_inf = json.loads(out_inf.read_text(encoding="utf-8"))["corpus_calibration"]
    assert data_inf["max_index_frequency"] is None
    assert data_inf["total_units"] == 50

    # 3. Valid float should be cleanly preserved
    calib_valid = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "shingle_frequencies": {"k": 10},
    }
    out_valid = tmp_path / "baseline_valid.json"
    record_baseline(clones, str(out_valid), "target_repo", 0.90, corpus_calibration=calib_valid)
    data_valid = json.loads(out_valid.read_text(encoding="utf-8"))["corpus_calibration"]
    assert data_valid["max_index_frequency"] == 0.25
    assert data_valid["total_units"] == 100

    loaded = load_baseline(str(out_valid))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["total_units"] == 100
    assert loaded.corpus_calibration["max_index_frequency"] == 0.25

def test_diff_scan_uncalibrated_pruning(tmp_path: Path) -> None:
    """Test diff-only scan without calibration prunes ubiquitous boilerplate across repository units."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_pruning"
    repo.mkdir()

    boilerplate_code = (
        "def common_logging_handler(event_name, payload, context):\n"
        "    tag = 'AUDIT'\n"
        "    msg = f'[{tag}] {event_name}: {payload}'\n"
        "    print(msg)\n"
        "    log_line = {'event': event_name, 'ctx': context}\n"
        "    return log_line\n"
    )

    for i in range(34):
        (repo / f"unmod_{i}.py").write_text(boilerplate_code, encoding="utf-8")

    mod_code = (
        boilerplate_code + "\n\n"
        "def calculate_tax_for_eu_region(amount, rate, country_code):\n"
        "    subtotal = amount * rate\n"
        "    tax = subtotal * 0.20\n"
        "    total = subtotal + tax\n"
        "    status = 'EU_OK'\n"
        "    return {'total': total, 'tax': tax, 'status': status}\n"
    )
    (repo / "mod.py").write_text(mod_code, encoding="utf-8")

    target_code = (
        "def calculate_tax_for_eu_region_duplicate(amount, rate, country_code):\n"
        "    subtotal = amount * rate\n"
        "    tax = subtotal * 0.20\n"
        "    total = subtotal + tax\n"
        "    status = 'EU_OK'\n"
        "    return {'total': total, 'tax': tax, 'status': status}\n"
    )
    (repo / "clone_target.py").write_text(target_code, encoding="utf-8")

    diff_clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["mod.py"],
        max_index_frequency=0.25,
        min_corpus_size=30,
        corpus_calibration=None,
    )

    assert len(diff_clones) == 1
    pair = diff_clones[0]
    matched_files = {Path(pair[1]["file"]).name, Path(pair[2]["file"]).name}
    assert matched_files == {"mod.py", "clone_target.py"}
    assert pair[1]["name"] in ("calculate_tax_for_eu_region", "calculate_tax_for_eu_region_duplicate")

def test_diff_scan_avoids_pruning_novel_shingles_missing_from_calibration(tmp_path: Path) -> None:
    """Verifies that shingles missing from calibration frequencies in differential mode are not falsely pruned."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.parser import harvest_file_units  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_stale_baseline"
    repo.mkdir()

    def generate_fn(num: int) -> str:
        return (
            f"def custom_domain_service_action_{num}(payload, context, options):\n"
            "    transformed = [item.strip() for item in payload if item]\n"
            "    result_map = {idx: val.upper() for idx, val in enumerate(transformed)}\n"
            f"    audit_tag = 'ACTION_{num}'\n"
            "    status_flag = len(result_map) > 0\n"
            "    return {'status': status_flag, 'tag': audit_tag, 'data': result_map}\n"
        )

    diff_code_1 = "\n\n".join(generate_fn(i) for i in range(4))
    diff_code_2 = "\n\n".join(generate_fn(i) for i in range(4))
    (repo / "new_service_a.py").write_text(diff_code_1, encoding="utf-8")
    (repo / "new_service_b.py").write_text(diff_code_2, encoding="utf-8")

    # Stale/small baseline calibration with total_units=1 and empty shingle_frequencies:
    # total_corpus_units = 1 + 8 = 9.
    # With max_index_frequency=0.25 and effective_min_corpus=4, max_posting_len = max(2, ceil(9 * 0.25)) = 3.
    # The novel domain shingles occur in 8 units (or 4 per file).
    # If uncalibrated df_local=8 (or 4) were compared with max_posting_len=3, it would be falsely pruned!
    # Because it is missing from calibration frequencies, the posting-length check is bypassed.
    calib = {
        "total_units": 1,
        "max_index_frequency": 0.25,
        "min_lines": 6,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }

    clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["new_service_a.py", "new_service_b.py"],
        corpus_calibration=calib,
    )

    cross_file_clones = [
        (sim, u1, u2)
        for sim, u1, u2 in clones
        if Path(u1["file"]).name != Path(u2["file"]).name
    ]
    assert len(cross_file_clones) >= 4
    for _, u1, u2 in cross_file_clones:
        f1 = Path(u1["file"]).name
        f2 = Path(u2["file"]).name
        assert {f1, f2} == {"new_service_a.py", "new_service_b.py"}

    # Verify that global_stop_shingles is STILL honored for novel shingles if present in global_stop_shingles
    units = harvest_file_units(str(repo / "new_service_a.py"), str(repo), min_lines=6)
    assert len(units) >= 1

    calib_all_stopped = {
        "total_units": 1,
        "max_index_frequency": 0.25,
        "min_lines": 6,
        "min_corpus_size": 4,
        "global_stop_shingles": set(units[0]["shingles"]),
        "shingle_frequencies": {},
    }
    clones_all_stopped = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["new_service_a.py", "new_service_b.py"],
        corpus_calibration=calib_all_stopped,
    )
    names = {u1["name"] for _, u1, _ in clones_all_stopped}
    assert "custom_domain_service_action_0" not in names

def test_corpus_calibration_malformed_types_and_unserializable_shingles(tmp_path: Path) -> None:
    """Verifies that non-dict calibration, non-dict shingle_frequencies, and un-serializable objects do not crash."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import record_baseline  # pylint: disable=import-outside-toplevel

    code_file = tmp_path / "sample.py"
    code_file.write_text(
        "def compute_something(a, b):\n"
        "    x = a * 2 + b\n"
        "    y = x ** 2 - 1\n"
        "    return y if y > 0 else 0\n\n"
        "def compute_something_dup(a, b):\n"
        "    x = a * 2 + b\n"
        "    y = x ** 2 - 1\n"
        "    return y if y > 0 else 0\n",
        encoding="utf-8",
    )

    # 1. Non-dict corpus_calibration passed to scan_target
    clones = scan_target(str(tmp_path), min_lines=3, corpus_calibration="not_a_dict")  # type: ignore[arg-type]
    assert len(clones) >= 1

    # 2. corpus_calibration with shingle_frequencies=None under tfidf=True
    clones_tfidf = scan_target(
        str(tmp_path),
        min_lines=3,
        tfidf=True,
        corpus_calibration={"shingle_frequencies": None, "total_units": 5},
    )
    assert len(clones_tfidf) >= 1

    # 3. corpus_calibration with non-dict shingle_frequencies
    clones_bad_freqs = scan_target(
        str(tmp_path),
        min_lines=3,
        corpus_calibration={"shingle_frequencies": "not_a_dict", "total_units": 5},
    )
    assert len(clones_bad_freqs) >= 1

    # 4. record_baseline with non-dict corpus_calibration
    base_file = tmp_path / "base1.json"
    record_baseline(
        clones,
        str(base_file),
        str(tmp_path),
        0.9,
        corpus_calibration="not_a_dict",  # type: ignore[arg-type]
    )
    assert base_file.exists()

    # 5. record_baseline with un-JSON-serializable objects in global_stop_shingles
    base_file_unserializable = tmp_path / "base2.json"
    record_baseline(
        clones,
        str(base_file_unserializable),
        str(tmp_path),
        0.9,
        corpus_calibration={
            "total_units": 10,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [({1, 2}, [object()])],
            "shingle_frequencies": {"shingle_key_1": 5},
        },
    )
    assert base_file_unserializable.exists()

    # 6. tfidf=True with non-finite and negative frequencies in corpus_calibration
    clones_nonfinite = scan_target(
        str(tmp_path),
        min_lines=3,
        tfidf=True,
        corpus_calibration={
            "total_units": 5,
            "shingle_frequencies": {
                "compute_something": float("inf"),
                "var_key": float("-inf"),
                "other_key": float("nan"),
                "neg_key": -999,
                "str_inf": "inf",
            },
        },
    )
    assert len(clones_nonfinite) >= 1

    # 7. candidate pairing with non-finite and negative frequencies in diff and normal mode
    clones_diff_nonfinite = scan_target(
        str(tmp_path),
        min_lines=3,
        diff_files=["sample.py"],
        corpus_calibration={
            "total_units": 5,
            "max_index_frequency": 0.25,
            "shingle_frequencies": {
                "compute_something": float("inf"),
                "neg_key": -50,
            },
        },
    )
    assert len(clones_diff_nonfinite) >= 1

def test_type_tagged_shingle_keys_and_oversized_unit_bounds(tmp_path: Path) -> None:
    """Verifies type-tagged shingle keys, legacy key backward compatibility, and oversized unit bounds."""
    from pydoppelgangerhunt._baseline_calibration import (  # pylint: disable=import-outside-toplevel
        _deserialize_shingle_key,
        _safe_total_units,
        _serialize_shingle_key,
    )
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        MAX_CALIBRATION_UNITS,
        load_baseline as load_base,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import _lookup_calib_freq, scan_target  # pylint: disable=import-outside-toplevel

    # 1. Type-tagged key serialization & deserialization fidelity
    # Tuples
    tup = ("Module", "If", "Compare")
    assert _serialize_shingle_key(tup) == 't:["Module", "If", "Compare"]'
    assert _deserialize_shingle_key('t:["Module", "If", "Compare"]') == tup

    # Scalar strings (including ones that look like JSON lists)
    s_like_list = '["Module", "If"]'
    assert _serialize_shingle_key(s_like_list) == 's:["Module", "If"]'
    assert _deserialize_shingle_key('s:["Module", "If"]') == s_like_list
    assert isinstance(_deserialize_shingle_key('s:["Module", "If"]'), str)

    # Strings with prefixes
    s_prefix = "t:custom_tag"
    assert _serialize_shingle_key(s_prefix) == "s:t:custom_tag"
    assert _deserialize_shingle_key("s:t:custom_tag") == s_prefix

    # Numbers and booleans
    assert _serialize_shingle_key(42) == "i:42"
    assert _deserialize_shingle_key("i:42") == 42
    assert _serialize_shingle_key(3.14) == "f:3.14"
    assert _deserialize_shingle_key("f:3.14") == 3.14
    assert _serialize_shingle_key(True) == "b:1"
    assert _deserialize_shingle_key("b:1") is True

    # Legacy untagged format compatibility
    assert _deserialize_shingle_key('["Legacy", "Shingle"]') == ("Legacy", "Shingle")
    assert _deserialize_shingle_key("plain_legacy_str") == "plain_legacy_str"

    # 2. _safe_total_units bounds
    assert _safe_total_units(100) == 100
    assert _safe_total_units(0) == 0
    assert _safe_total_units(-50) == 0
    assert _safe_total_units(None) == 0
    assert _safe_total_units("invalid") == 0
    assert _safe_total_units(10**400) == MAX_CALIBRATION_UNITS
    assert _safe_total_units(MAX_CALIBRATION_UNITS + 500) == MAX_CALIBRATION_UNITS

    # 3. _lookup_calib_freq multi-representation lookup
    freqs = {
        ("Exact", "Tuple"): 10,
        't:["Tagged", "Tuple"]': 20,
        '["Legacy", "Tuple"]': 30,
        "s:tagged_string": 40,
        "exact_string": 50,
        "i:99": 60,
    }
    assert _lookup_calib_freq(freqs, ("Exact", "Tuple")) == 10
    assert _lookup_calib_freq(freqs, ("Tagged", "Tuple")) == 20
    assert _lookup_calib_freq(freqs, ("Legacy", "Tuple")) == 30
    assert _lookup_calib_freq(freqs, "tagged_string") == 40
    assert _lookup_calib_freq(freqs, "exact_string") == 50
    assert _lookup_calib_freq(freqs, 99) == 60
    assert _lookup_calib_freq(freqs, "missing") is None

    # 4. TF-IDF and candidate pairing with oversized total_units
    code_file = tmp_path / "oversized_sample.py"
    code_file.write_text(
        "def oversized_test_fn(x, y):\n"
        "    res = x * 3 + y\n"
        "    res2 = res ** 2 + 10\n"
        "    return res2 if res2 > 5 else 0\n\n"
        "def oversized_test_fn_dup(x, y):\n"
        "    res = x * 3 + y\n"
        "    res2 = res ** 2 + 10\n"
        "    return res2 if res2 > 5 else 0\n",
        encoding="utf-8",
    )

    clones_oversized = scan_target(
        str(tmp_path),
        min_lines=3,
        tfidf=True,
        corpus_calibration={
            "total_units": 10**400,
            "shingle_frequencies": {},
        },
    )
    assert len(clones_oversized) >= 1

    # 5. Global frequencies validated and clamped to total_units
    clones_freq_clamped = scan_target(
        str(tmp_path),
        min_lines=3,
        tfidf=True,
        corpus_calibration={
            "total_units": 10,
            "shingle_frequencies": {
                "oversized_test_fn": 99999,  # Exceeds total_units=10
            },
        },
    )
    assert len(clones_freq_clamped) >= 1

    # 6. Candidate pairing with total_units=0 and frequencies > 0 does not prune candidates
    clones_zero_units = scan_target(
        str(tmp_path),
        min_lines=3,
        corpus_calibration={
            "total_units": 0,
            "max_index_frequency": 0.25,
            "shingle_frequencies": {
                "oversized_test_fn": 100,
            },
        },
    )
    assert len(clones_zero_units) >= 1

    # 7. record_baseline and load_baseline round-trip with type-tagged keys and oversized bounds
    base_file = tmp_path / "tagged_baseline.json"
    record_baseline(
        clones_oversized,
        str(base_file),
        str(tmp_path),
        0.90,
        corpus_calibration={
            "total_units": 10**400,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [("A", "B"), '["string_that_looks_like_tuple"]', 123],
            "shingle_frequencies": {
                ("A", "B"): 5,
                '["string_that_looks_like_tuple"]': 10,
                123: 15,
            },
        },
    )
    raw_saved = json.loads(base_file.read_text(encoding="utf-8"))
    calib_saved = raw_saved["corpus_calibration"]
    assert calib_saved["total_units"] == MAX_CALIBRATION_UNITS
    assert 't:["A", "B"]' in calib_saved["shingle_frequencies"]
    assert 's:["string_that_looks_like_tuple"]' in calib_saved["shingle_frequencies"]
    assert "i:123" in calib_saved["shingle_frequencies"]

    loaded = load_base(str(base_file))
    assert loaded.corpus_calibration is not None
    loaded_calib = loaded.corpus_calibration
    assert loaded_calib["total_units"] == MAX_CALIBRATION_UNITS
    assert loaded_calib["shingle_frequencies"][("A", "B")] == 5
    assert loaded_calib["shingle_frequencies"]['["string_that_looks_like_tuple"]'] == 10
    assert loaded_calib["shingle_frequencies"][123] == 15

def test_diff_scan_prunes_uncalibrated_shingles_matching_many_unmodified_units(tmp_path: Path) -> None:
    """Verifies that uncalibrated shingles matching many unmodified units are bounded/pruned to prevent pair explosion."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_explosion_defense"
    repo.mkdir()

    # Create 10 unmodified files, each containing a common helper with identical ubiquitous structure
    common_code = (
        "def common_repository_helper(context, data):\n"
        "    if not context:\n"
        "        return None\n"
        "    res = {}\n"
        "    for k, v in data.items():\n"
        "        res[k] = str(v).strip()\n"
        "    return res\n"
    )
    for i in range(10):
        (repo / f"unmodified_{i}.py").write_text(
            f"# File unmodified {i}\n" + common_code,
            encoding="utf-8",
        )

    # Create 1 changed file in the diff containing the same common helper plus a unique function
    changed_code = (
        common_code
        + "\n\ndef unique_changed_feature():\n"
        + "    return 'UNIQUE_DIFF_FEATURE_ALPHA_BETA'\n"
    )
    (repo / "changed.py").write_text(changed_code, encoding="utf-8")

    # Stale/partial calibration missing common_repository_helper shingles entirely:
    # 11 total units in repo (10 unmodified + 1 changed).
    # calib specifies total_units=10, max_index_frequency=0.25, min_corpus_size=4.
    # Because common_repository_helper occurs in 10 unmodified units, df_unmodified = 10 > max_posting_len (which is ceil(11 * 0.25) = 3).
    # The common shingle MUST be pruned from pairing against the 10 unmodified units!
    calib = {
        "total_units": 10,
        "max_index_frequency": 0.25,
        "min_lines": 6,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }

    clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.85,
        diff_files=["changed.py"],
        corpus_calibration=calib,
    )

    # The changed file should NOT be paired with all 10 unmodified files across the common helper
    changed_clones = [
        (sim, u1, u2)
        for sim, u1, u2 in clones
        if "changed.py" in (Path(u1["file"]).name, Path(u2["file"]).name)
    ]
    assert len(changed_clones) == 0

    # Also verify that if two diff files share a novel domain clone (df_unmodified == 0), it IS detected
    novel_clone_code = (
        "def novel_domain_handler(request, response):\n"
        "    data_items = [x for x in request if x]\n"
        "    mapped = {i: str(v).lower() for i, v in enumerate(data_items)}\n"
        "    audit = 'DOMAIN_HANDLER_SECURE_TAG'\n"
        "    return {'audit': audit, 'mapped': mapped}\n"
    )
    (repo / "changed_a.py").write_text(novel_clone_code, encoding="utf-8")
    (repo / "changed_b.py").write_text(novel_clone_code, encoding="utf-8")

    novel_clones = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        diff_files=["changed_a.py", "changed_b.py"],
        corpus_calibration=dict(calib, min_lines=5),
    )
    cross_diff_clones = [
        (sim, u1, u2)
        for sim, u1, u2 in novel_clones
        if {Path(u1["file"]).name, Path(u2["file"]).name} == {"changed_a.py", "changed_b.py"}
    ]
    assert len(cross_diff_clones) >= 1

def test_min_corpus_size_malformed_fallback(tmp_path: Path) -> None:
    """Verifies that non-integer, negative, or invalid min_corpus_size safely falls back without raising errors."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel

    code_file = tmp_path / "min_corpus_sample.py"
    code_file.write_text(
        "def helper_sample_one(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n\n"
        "def helper_sample_two(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n",
        encoding="utf-8",
    )

    # 1. Invalid min_corpus_size in corpus_calibration
    clones = scan_target(
        str(tmp_path),
        min_lines=2,
        corpus_calibration={"min_corpus_size": "invalid", "total_units": 10},  # type: ignore[dict-item]
    )
    assert len(clones) >= 1

    # 2. Negative min_corpus_size in corpus_calibration
    clones_neg = scan_target(
        str(tmp_path),
        min_lines=2,
        corpus_calibration={"min_corpus_size": -50, "total_units": 10},
    )
    assert len(clones_neg) >= 1

    # 3. Direct invalid min_corpus_size parameter in scan_target
    clones_arg = scan_target(
        str(tmp_path),
        min_lines=2,
        min_corpus_size="invalid",  # type: ignore[arg-type]
    )
    assert len(clones_arg) >= 1

    # 4. compute_corpus_calibration with invalid min_corpus_size
    calib = compute_corpus_calibration([], min_corpus_size="invalid")  # type: ignore[arg-type]
    assert isinstance(calib, dict)

def test_corpus_calibration_feature_mode_validation_and_rejection(tmp_path: Path) -> None:
    """Verifies that feature space modes (bag_of_tokens, call_sequences) are tracked and mismatched calibration is skipped."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    code_file = tmp_path / "feature_mode_sample.py"
    code_file.write_text(
        "def func_alpha(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n\n"
        "def func_beta(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n",
        encoding="utf-8",
    )

    # 1. Feature flags are properly recorded
    calib_bot = compute_corpus_calibration([], bag_of_tokens=True)
    assert calib_bot["bag_of_tokens"] is True
    assert calib_bot["call_sequences"] is False

    calib_calls = compute_corpus_calibration([], call_sequences=True)
    assert calib_calls["bag_of_tokens"] is False
    assert calib_calls["call_sequences"] is True

    calib_default = compute_corpus_calibration([])
    assert calib_default["bag_of_tokens"] is False
    assert calib_default["call_sequences"] is False

    # 2. Incompatible calibration mode is skipped in scan_target without corrupting AST clone detection
    # Passing bag_of_tokens=True calibration to a standard AST shingle scan
    clones = scan_target(
        str(tmp_path),
        min_lines=2,
        corpus_calibration=calib_bot,
        bag_of_tokens=False,
        call_sequences=False,
    )
    # The clone must be detected and not falsely pruned by incompatible calibration lookups
    assert len(clones) >= 1

def test_malformed_crafted_calibration_metadata(tmp_path: Path) -> None:
    """Verifies that non-boolean and crafted strings in calibration metadata are safely coerced and sanitized."""
    from pydoppelgangerhunt._baseline_calibration import (  # pylint: disable=import-outside-toplevel
        _attach_calibration_flags,
        _safe_bool,
    )
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        load_baseline,
    )
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    # 1. _safe_bool coercion checks
    assert _safe_bool(True) is True
    assert _safe_bool(False) is False
    assert _safe_bool(1) is True
    assert _safe_bool(0) is False
    assert _safe_bool("true") is True
    assert _safe_bool("True") is True
    assert _safe_bool("1") is True
    assert _safe_bool("yes") is True
    assert _safe_bool("false") is False
    assert _safe_bool("False") is False
    assert _safe_bool("0") is False
    assert _safe_bool("no") is False
    assert _safe_bool(["invalid"]) is False
    assert _safe_bool({"key": "val"}) is False
    assert _safe_bool(None) is False

    # 2. _attach_calibration_flags with malformed types
    target_dict: Dict[str, Any] = {}
    _attach_calibration_flags(target_dict, "not_a_dict")  # type: ignore[arg-type]
    assert not target_dict

    crafted_source = {
        "bag_of_tokens": "false",
        "call_sequences": "0",
        "filter_stop_shingles": ["invalid"],
        "min_corpus_size": "crafted_string",
    }
    _attach_calibration_flags(target_dict, crafted_source)
    assert target_dict["bag_of_tokens"] is False
    assert target_dict["call_sequences"] is False
    assert target_dict["filter_stop_shingles"] is False
    assert target_dict["min_corpus_size"] is None

    # Negative min_corpus_size sanitized to None
    _attach_calibration_flags(target_dict, {"min_corpus_size": -42})
    assert target_dict["min_corpus_size"] is None

    # 3. Compatibility checking on malformed calib objects
    assert _is_calibration_mode_compatible("not_a_dict", bag_of_tokens=False, call_sequences=False) is False  # type: ignore[arg-type]
    assert _is_calibration_mode_compatible({"bag_of_tokens": "false"}, bag_of_tokens=False, call_sequences=False) is True
    assert _is_calibration_mode_compatible({"bag_of_tokens": "true"}, bag_of_tokens=True, call_sequences=False) is True

    # 4. JSON loading with spoofed fields
    base_file = tmp_path / "spoofed_baseline.json"
    spoofed_json = {
        "version": "1.4.0",
        "target": "repo",
        "threshold": 0.85,
        "clone_count": 0,
        "fingerprints": [],
        "corpus_calibration": {
            "total_units": 50,
            "max_index_frequency": 0.25,
            "bag_of_tokens": "false",
            "call_sequences": ["crafted"],
            "min_corpus_size": "invalid_size",
        },
    }
    base_file.write_text(json.dumps(spoofed_json), encoding="utf-8")
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["bag_of_tokens"] is False
    assert loaded.corpus_calibration["call_sequences"] is False
    assert loaded.corpus_calibration["min_corpus_size"] is None

def test_stop_shingles_calibration_compatibility_and_isolation(tmp_path: Path) -> None:
    """Verifies that configuration-derived stops are kept out of calibration and unconfigured scans reject stop-shingle calibration."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import (  # pylint: disable=import-outside-toplevel
        DEFAULT_STOP_SHINGLES,
        _is_calibration_mode_compatible,
        scan_target,
    )

    # 1. compute_corpus_calibration does not inject DEFAULT_STOP_SHINGLES into global_stop_shingles
    calib = compute_corpus_calibration([], filter_stop_shingles=True, stop_shingles={("Custom", "Stop")})
    assert len(calib["global_stop_shingles"]) == 0
    for s in DEFAULT_STOP_SHINGLES:
        assert s not in calib["global_stop_shingles"]
    assert ("Custom", "Stop") not in calib["global_stop_shingles"]
    assert calib["filter_stop_shingles"] is True

    # 2. _is_calibration_mode_compatible rejects filter_stop_shingles true-to-false transition
    assert _is_calibration_mode_compatible(calib, bag_of_tokens=False, call_sequences=False, filter_stop_shingles=True) is True
    assert _is_calibration_mode_compatible(calib, bag_of_tokens=False, call_sequences=False, filter_stop_shingles=False) is False

    # 3. scan_target skips calibration recorded with filter_stop_shingles=True when current scan has filter_stop_shingles=False
    code_file = tmp_path / "stop_shingle_test.py"
    code_file.write_text(
        "def fn_a(a, b):\n"
        "    x = a + b\n"
        "    y = x * 2\n"
        "    return y\n\n"
        "def fn_b(a, b):\n"
        "    x = a + b\n"
        "    y = x * 2\n"
        "    return y\n",
        encoding="utf-8",
    )
    clones = scan_target(
        str(tmp_path),
        min_lines=2,
        min_tokens=3,
        corpus_calibration=calib,
        filter_stop_shingles=False,
    )
    assert len(clones) >= 1

def test_harvesting_modes_calibration_persistence_and_compatibility(tmp_path: Path) -> None:
    """Verifies that AST unit-shaping and harvesting modes are persisted in calibration and validated for compatibility."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        HARVEST_BOOLEAN_MODES,
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import (  # pylint: disable=import-outside-toplevel
        _is_calibration_mode_compatible,
        scan_target,
    )

    # 1. compute_corpus_calibration attaches all default harvesting boolean modes and int bounds
    calib_default = compute_corpus_calibration([])
    for flag, default_val in HARVEST_BOOLEAN_MODES:
        assert flag in calib_default
        assert calib_default[flag] is default_val
    assert calib_default["window_size"] == 5
    assert calib_default["min_expr_complexity"] == 4

    # 2. compute_corpus_calibration overrides harvesting flags when explicitly provided
    calib_custom = compute_corpus_calibration(
        [],
        blind_literals=True,
        strip_annotations=False,
        idioms=True,
        commutative=True,
        filter_boilerplate=True,
        sliding_window=True,
        window_size=10,
        complex_expressions=True,
        min_expr_complexity=8,
    )
    assert calib_custom["blind_literals"] is True
    assert calib_custom["strip_annotations"] is False
    assert calib_custom["idioms"] is True
    assert calib_custom["commutative"] is True
    assert calib_custom["filter_boilerplate"] is True
    assert calib_custom["sliding_window"] is True
    assert calib_custom["window_size"] == 10
    assert calib_custom["complex_expressions"] is True
    assert calib_custom["min_expr_complexity"] == 8

    # 3. Round-trip baseline persistence through record_baseline and load_baseline
    base_file = tmp_path / "harvest_baseline.json"
    record_baseline([], str(base_file), "target", 0.90, corpus_calibration=calib_custom)
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    loaded_calib = loaded.corpus_calibration
    assert loaded_calib["blind_literals"] is True
    assert loaded_calib["strip_annotations"] is False
    assert loaded_calib["idioms"] is True
    assert loaded_calib["commutative"] is True
    assert loaded_calib["filter_boilerplate"] is True
    assert loaded_calib["sliding_window"] is True
    assert loaded_calib["window_size"] == 10
    assert loaded_calib["complex_expressions"] is True
    assert loaded_calib["min_expr_complexity"] == 8

    # 4. _is_calibration_mode_compatible rejects mismatched harvesting configurations
    assert _is_calibration_mode_compatible(
        calib_custom,
        blind_literals=True,
        strip_annotations=False,
        idioms=True,
        commutative=True,
        filter_boilerplate=True,
        sliding_window=True,
        window_size=10,
        complex_expressions=True,
        min_expr_complexity=8,
    ) is True

    # Mismatched blind_literals
    assert _is_calibration_mode_compatible(calib_custom, blind_literals=False) is False
    assert _is_calibration_mode_compatible(calib_default, blind_literals=True) is False

    # Mismatched strip_annotations
    assert _is_calibration_mode_compatible(calib_custom, strip_annotations=True) is False
    assert _is_calibration_mode_compatible(calib_default, strip_annotations=False) is False

    # Mismatched idioms
    assert _is_calibration_mode_compatible(calib_custom, idioms=False) is False
    assert _is_calibration_mode_compatible(calib_default, idioms=True) is False

    # Mismatched commutative
    assert _is_calibration_mode_compatible(calib_custom, commutative=False) is False
    assert _is_calibration_mode_compatible(calib_default, commutative=True) is False

    # Mismatched filter_boilerplate
    assert _is_calibration_mode_compatible(calib_custom, filter_boilerplate=False) is False
    assert _is_calibration_mode_compatible(calib_default, filter_boilerplate=True) is False

    # Mismatched window_size when sliding_window is active
    assert _is_calibration_mode_compatible(
        calib_custom,
        blind_literals=True,
        strip_annotations=False,
        idioms=True,
        commutative=True,
        filter_boilerplate=True,
        sliding_window=True,
        window_size=5,
        complex_expressions=True,
        min_expr_complexity=8,
    ) is False

    # Mismatched min_expr_complexity when complex_expressions is active
    assert _is_calibration_mode_compatible(
        calib_custom,
        blind_literals=True,
        strip_annotations=False,
        idioms=True,
        commutative=True,
        filter_boilerplate=True,
        sliding_window=True,
        window_size=10,
        complex_expressions=True,
        min_expr_complexity=4,
    ) is False

    # 5. Integration: scan_target skips incompatible calibration
    code_file = tmp_path / "diff_test.py"
    code_file.write_text(
        "def service_one(a, b):\n"
        "    v1 = a * 10\n"
        "    v2 = b + 20\n"
        "    return v1 + v2\n\n"
        "def service_two(a, b):\n"
        "    v1 = a * 10\n"
        "    v2 = b + 20\n"
        "    return v1 + v2\n",
        encoding="utf-8",
    )
    clones = scan_target(
        str(tmp_path),
        min_lines=2,
        min_tokens=3,
        corpus_calibration=calib_custom,
        blind_literals=False,
    )
    assert len(clones) >= 1

def test_calibration_bounds_and_frequency_cutoff_compatibility(tmp_path: Path) -> None:
    """Verifies that min_lines, min_tokens, and max_index_frequency are persisted and validated for compatibility."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import (  # pylint: disable=import-outside-toplevel
        _is_calibration_mode_compatible,
        scan_target,
    )

    # 1. Default and custom bounds in compute_corpus_calibration
    c_default = compute_corpus_calibration([])
    assert c_default["min_lines"] == 8
    assert c_default["min_tokens"] == 15
    assert c_default["max_index_frequency"] == 0.25

    c_custom = compute_corpus_calibration(
        [],
        min_lines=6,
        min_tokens=10,
        max_index_frequency=0.40,
    )
    assert c_custom["min_lines"] == 6
    assert c_custom["min_tokens"] == 10
    assert c_custom["max_index_frequency"] == 0.40

    # 2. Round-trip baseline persistence
    base_file = tmp_path / "bounds_baseline.json"
    record_baseline([], str(base_file), "target", 0.90, corpus_calibration=c_custom)
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["min_lines"] == 6
    assert loaded.corpus_calibration["min_tokens"] == 10
    assert loaded.corpus_calibration["max_index_frequency"] == 0.40

    # 3. Direct compatibility check for min_lines and min_tokens
    assert _is_calibration_mode_compatible(c_custom, min_lines=6, min_tokens=10, max_index_frequency=0.40) is True
    assert _is_calibration_mode_compatible(c_custom, min_lines=8, min_tokens=10, max_index_frequency=0.40) is False
    assert _is_calibration_mode_compatible(c_custom, min_lines=6, min_tokens=15, max_index_frequency=0.40) is False
    assert _is_calibration_mode_compatible(c_default, min_lines=6, min_tokens=15, max_index_frequency=0.25) is False
    assert _is_calibration_mode_compatible(c_default, min_lines=8, min_tokens=10, max_index_frequency=0.25) is False

    # 4. Direct compatibility check for max_index_frequency
    assert _is_calibration_mode_compatible(c_default, max_index_frequency=0.50) is False
    assert _is_calibration_mode_compatible(c_default, max_index_frequency=None) is False

    c_none = compute_corpus_calibration([], max_index_frequency=None)
    assert c_none["max_index_frequency"] is None
    assert _is_calibration_mode_compatible(c_none, max_index_frequency=None) is True
    assert _is_calibration_mode_compatible(c_none, max_index_frequency=0.25) is False

    # 5. Integration: Lowered min_lines threshold in diff scan skips calibration
    repo_dir = tmp_path / "threshold_repo"
    repo_dir.mkdir()
    fn_7_lines = (
        "def micro_service_step(val_x, val_y):\n"
        "    t1 = val_x * 3\n"
        "    t2 = val_y * 7\n"
        "    t3 = t1 + t2\n"
        "    t4 = t3 * 2\n"
        "    t5 = t4 - 1\n"
        "    return t5 if t5 > 0 else 0\n"
    )
    (repo_dir / "worker_a.py").write_text(fn_7_lines, encoding="utf-8")
    (repo_dir / "worker_b.py").write_text(fn_7_lines, encoding="utf-8")

    # Baseline recorded with default min_lines=8
    calib_base_8 = compute_corpus_calibration([], min_lines=8)
    # Scanning with min_lines=6 lowers the threshold compared to calib_base_8
    clones_lowered = scan_target(
        str(repo_dir),
        min_lines=6,
        min_tokens=5,
        threshold=0.90,
        corpus_calibration=calib_base_8,
    )
    # Incompatible calibration is bypassed and the 7-line clone is detected
    assert len(clones_lowered) == 1
    # When scanned with matching min_lines=8, the 7-line function is naturally filtered out
    clones_strict = scan_target(
        str(repo_dir),
        min_lines=8,
        threshold=0.90,
        corpus_calibration=calib_base_8,
    )
    assert len(clones_strict) == 0

    # 6. min_corpus_size compatibility and scan-argument authoritativeness
    c_mcs_4 = compute_corpus_calibration([], min_corpus_size=4)
    assert _is_calibration_mode_compatible(c_mcs_4, min_corpus_size=100) is False
    assert _is_calibration_mode_compatible(c_mcs_4, min_corpus_size=4) is True
    assert _is_calibration_mode_compatible(c_mcs_4, min_corpus_size=None) is True

    clones_mcs_override = scan_target(
        str(repo_dir),
        min_lines=6,
        min_tokens=5,
        min_corpus_size=100,
        corpus_calibration=c_mcs_4,
    )
    assert len(clones_mcs_override) == 1

def test_calibrated_differential_tfidf_weights_unchanged_candidate_partner_shingles(
    tmp_path: Path,
) -> None:
    """Verifies calibrated differential TF-IDF weights incorporate global frequencies for unchanged partner shingles."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    f_changed = tmp_path / "changed.py"
    f_unchanged = tmp_path / "unchanged.py"

    code_changed = (
        "def run_pipeline(x):\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    return x + a + b + c + d\n"
    )
    # Extra statement in unchanged creates shingles present ONLY in unchanged.py
    code_unchanged = (
        "def run_pipeline(x):\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    extra_unique_val = 99999 + 88888\n"
        "    return x + a + b + c + d\n"
    )

    f_changed.write_text(code_changed, encoding="utf-8")
    f_unchanged.write_text(code_unchanged, encoding="utf-8")

    # 1. Harvest units to find shingles unique to unchanged
    units_raw = scan_target(str(tmp_path), min_lines=5, threshold=0.60)
    assert len(units_raw) == 1

    # 2. Build calibration with high global frequency for all shingles
    all_units, calib_dict = scan_target(
        str(tmp_path), min_lines=5, threshold=0.60, return_calibration=True
    )
    assert isinstance(calib_dict, dict)
    assert len(all_units) == 1

    # Modify calibration to simulate a large corpus where shingles have moderate global count (10% < 25%)
    calib_dict["total_units"] = 500
    freq_map = calib_dict.get("shingle_frequencies", {})
    for sh in list(freq_map.keys()):
        freq_map[sh] = 50

    # 3. Differential scan where only changed.py is in diff_files
    diff_clones = scan_target(
        str(tmp_path),
        min_lines=5,
        threshold=0.60,
        tfidf=True,
        diff_files=["changed.py"],
        corpus_calibration=calib_dict,
    )
    assert len(diff_clones) == 1
    sim, u1, u2 = diff_clones[0]
    assert sim >= 0.60
    assert {u1["file"], u2["file"]} == {"changed.py", "unchanged.py"}

def test_calibration_config_hash_determinism_and_sensitivity() -> None:
    """Verifies that compute_calibration_config_hash is deterministic and sensitive to all parameters."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel

    cfg1: Dict[str, Any] = {"bag_of_tokens": False, "min_lines": 8, "max_index_frequency": 0.25, "idioms": True}
    cfg2: Dict[str, Any] = {"idioms": True, "max_index_frequency": 0.25, "min_lines": 8, "bag_of_tokens": False}
    hash1 = compute_calibration_config_hash(cfg1)
    hash2 = compute_calibration_config_hash(cfg2)
    assert hash1 == hash2
    assert len(hash1) == 64

    # Sensitive to bag_of_tokens
    cfg_bot = dict(cfg1)
    cfg_bot["bag_of_tokens"] = True
    assert compute_calibration_config_hash(cfg_bot) != hash1

    # Sensitive to min_lines
    cfg_lines = dict(cfg1)
    cfg_lines["min_lines"] = 10
    assert compute_calibration_config_hash(cfg_lines) != hash1

    # Sensitive to max_index_frequency
    cfg_freq = dict(cfg1)
    cfg_freq["max_index_frequency"] = 0.50
    assert compute_calibration_config_hash(cfg_freq) != hash1

    # Sensitive to max_index_frequency=None
    cfg_none = dict(cfg1)
    cfg_none["max_index_frequency"] = None
    assert compute_calibration_config_hash(cfg_none) != hash1

    # Invariant to absent vs explicit None min_corpus_size
    assert compute_calibration_config_hash({}) == compute_calibration_config_hash({"min_corpus_size": None})
    assert compute_calibration_config_hash({"min_corpus_size": 100}) != compute_calibration_config_hash({"min_corpus_size": None})
    # Fallback to min_corpus_units when min_corpus_size is absent or None
    assert compute_calibration_config_hash({"min_corpus_units": 100}) == compute_calibration_config_hash({"min_corpus_size": 100})
    assert compute_calibration_config_hash({"min_corpus_size": None, "min_corpus_units": 100}) == compute_calibration_config_hash({"min_corpus_size": 100})
    # min_corpus_size takes precedence over min_corpus_units when both present
    assert compute_calibration_config_hash({"min_corpus_size": 50, "min_corpus_units": 100}) == compute_calibration_config_hash({"min_corpus_size": 50})
    # Preserves explicit min_corpus_size: 0
    assert compute_calibration_config_hash({"min_corpus_size": 0, "min_corpus_units": 100}) == compute_calibration_config_hash({"min_corpus_size": 0})
    assert compute_calibration_config_hash({"min_corpus_size": 0}) != compute_calibration_config_hash({"min_corpus_size": 100})

    from pydoppelgangerhunt._baseline_calibration import (  # pylint: disable=import-outside-toplevel
        _extract_calibration_settings,
    )
    assert _extract_calibration_settings({"min_corpus_units": 150})["min_corpus_size"] == 150
    assert _extract_calibration_settings({"min_corpus_size": 0, "min_corpus_units": 150})["min_corpus_size"] == 0
    # Fallback to exclude when excludes is absent or None
    assert _extract_calibration_settings({"exclude": ["vendor/*"]})["excludes"] == ["vendor/*"]
    assert _extract_calibration_settings({"excludes": None, "exclude": ["vendor/*"]})["excludes"] == ["vendor/*"]
    assert _extract_calibration_settings({"excludes": ["build/*"], "exclude": ["vendor/*"]})["excludes"] == ["build/*"]

    # Defensively handle explicit None in calls and vector during calibration
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel
    calib_calls_none = compute_corpus_calibration([{"calls": None}], call_sequences=True)
    assert calib_calls_none["total_units"] == 1
    calib_vector_none = compute_corpus_calibration([{"vector": None}], bag_of_tokens=True)
    assert calib_vector_none["total_units"] == 1
    # Defensively handle collection without .keys() in vector when bag_of_tokens is unset
    calib_vector_list = compute_corpus_calibration([{"vector": ["t1", "t2"]}], bag_of_tokens=False)
    assert calib_vector_list["total_units"] == 1
    assert "t1" in calib_vector_list["shingle_frequencies"]

def test_baseline_config_hash_and_recorded_commit_roundtrip(tmp_path: Path) -> None:
    """Verifies that record_baseline and load_baseline preserve config_hash and recorded_commit."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )

    baseline_file = str(tmp_path / "baseline_calib_hash.json")
    calib = compute_corpus_calibration([], min_lines=8, bag_of_tokens=True)
    assert "config_hash" in calib
    assert len(calib["config_hash"]) == 64

    saved_path = record_baseline([], baseline_file, str(tmp_path), 0.90, corpus_calibration=calib)
    assert Path(saved_path).exists()

    data = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert "config_hash" in data
    assert data["config_hash"] == calib["config_hash"]
    assert "corpus_calibration" in data
    assert data["corpus_calibration"]["config_hash"] == calib["config_hash"]

    loaded = load_baseline(baseline_file)
    assert loaded.config_hash == calib["config_hash"]
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration.get("config_hash") == calib["config_hash"]

def test_legacy_baseline_loads_without_config_hash_or_recorded_commit(tmp_path: Path) -> None:
    """Verifies that legacy baselines lacking config_hash or recorded_commit load safely."""
    from pydoppelgangerhunt.baseline import load_baseline  # pylint: disable=import-outside-toplevel

    legacy_file = tmp_path / "legacy_v14.json"
    legacy_data = {
        "version": "1.4.0",
        "threshold": 0.85,
        "clone_count": 0,
        "fingerprints": [],
        "corpus_calibration": {
            "total_units": 100,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [],
            "shingle_frequencies": {},
        },
    }
    legacy_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    loaded = load_baseline(str(legacy_file))
    assert loaded.corpus_calibration is not None
    assert "config_hash" in loaded.corpus_calibration
    assert loaded.recorded_commit is None

def test_baseline_metadata_type_validation_with_crafted_values(tmp_path: Path) -> None:
    """Verifies that non-string/numeric config_hash and recorded_commit degrade safely without raising in load_baseline."""
    crafted_path = tmp_path / "crafted_baseline.json"
    crafted_data = {
        "version": "1.5.0",
        "fingerprints": [],
        "config_hash": 123456789,
        "recorded_commit": 987654321,
        "corpus_calibration": {
            "total_units": 100,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [],
            "shingle_frequencies": {},
            "config_hash": 123456789,
            "recorded_commit": 987654321,
        },
    }
    crafted_path.write_text(json.dumps(crafted_data), encoding="utf-8")

    loaded = load_baseline(str(crafted_path))
    assert isinstance(loaded.config_hash, str)
    assert len(loaded.config_hash) == 64
    assert loaded.config_hash[:8]  # Can slice without TypeError
    assert loaded.recorded_commit is None  # Degraded safely from numeric value
    assert loaded.corpus_calibration is not None
    assert isinstance(loaded.corpus_calibration.get("config_hash"), str)
    assert loaded.corpus_calibration.get("recorded_commit") is None

def test_compute_calibration_config_hash_lossless_float() -> None:
    """Verifies that compute_calibration_config_hash hashes floats without lossy 6-decimal rounding."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel

    hash_a = compute_calibration_config_hash({"max_index_frequency": 0.25})
    hash_b = compute_calibration_config_hash({"max_index_frequency": 0.2500004})
    assert hash_a != hash_b

def test_compute_calibration_config_hash_malformed_frequency_fallback() -> None:
    """Verifies that unparseable max_index_frequency values fall back to 0.25 in config hash."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel

    hash_default = compute_calibration_config_hash({"max_index_frequency": 0.25})
    hash_invalid_str = compute_calibration_config_hash({"max_index_frequency": "invalid_frequency"})
    hash_invalid_bool = compute_calibration_config_hash({"max_index_frequency": True})
    hash_invalid_neg = compute_calibration_config_hash({"max_index_frequency": -0.5})
    assert hash_invalid_str == hash_default
    assert hash_invalid_bool == hash_default
    assert hash_invalid_neg == hash_default

    # Explicit None remains distinct
    hash_none = compute_calibration_config_hash({"max_index_frequency": None})
    assert hash_none != hash_default

def test_calibration_compatibility_audit_tests_and_include_notebooks() -> None:
    """Verifies that audit_tests and include_notebooks are validated in calibration compatibility."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    calib_std = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "audit_tests": False,
        "include_notebooks": False,
        "config_hash": compute_calibration_config_hash({"audit_tests": False, "include_notebooks": False}),
    }
    assert _is_calibration_mode_compatible(calib_std, audit_tests=False, include_notebooks=False)
    assert not _is_calibration_mode_compatible(calib_std, audit_tests=True, include_notebooks=False)
    assert not _is_calibration_mode_compatible(calib_std, audit_tests=False, include_notebooks=True)

def test_scan_target_full_scan_with_calibration_avoids_double_counting(tmp_path: Path) -> None:
    """Verifies that full scan with corpus_calibration does not double-count corpus units or frequencies."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    shared_code = (
        "def common_worker(x):\n"
        "    a = x + 1\n"
        "    b = a + 2\n"
        "    c = b + 3\n"
        "    d = c + 4\n"
        "    e = d + 5\n"
        "    f = e + 6\n"
        "    g = f + 7\n"
        "    return g\n"
    )
    for i in range(8):
        (tmp_path / f"shared_{i}.py").write_text(shared_code, encoding="utf-8")
    for i in range(22):
        unique_code = (
            f"def unique_{i}():\n"
            f"    alpha_{i} = 'hello'\n"
            f"    beta_{i} = 'world'\n"
            f"    gamma_{i} = 'foo'\n"
            f"    delta_{i} = 'bar'\n"
            f"    epsilon_{i} = 'baz'\n"
            f"    zeta_{i} = 'qux'\n"
            f"    eta_{i} = 'test'\n"
            f"    return alpha_{i} + beta_{i} + gamma_{i} + delta_{i} + epsilon_{i} + zeta_{i} + eta_{i}\n"
        )
        (tmp_path / f"unique_{i}.py").write_text(unique_code, encoding="utf-8")

    _, calib = scan_target(str(tmp_path), max_index_frequency=0.28, return_calibration=True)
    assert calib["total_units"] == 30

    clones_full = scan_target(
        str(tmp_path),
        max_index_frequency=0.28,
        corpus_calibration=calib,
        diff_files=None,
    )
    clones_uncalib = scan_target(
        str(tmp_path),
        max_index_frequency=0.28,
        corpus_calibration=None,
        diff_files=None,
    )
    assert len(clones_full) == 28
    assert len(clones_uncalib) == 28

def test_calibration_compatibility_with_excludes() -> None:
    """Verifies that excludes are preserved and compared during calibration compatibility checks."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    unit = {
        "file": "foo.py",
        "name": "bar",
        "shingles": ["sh1", "sh2"],
        "tokens": ["def", "bar"],
    }
    calib = compute_corpus_calibration([unit], excludes=["vendor", "tests"])
    assert calib.get("excludes") == ["tests", "vendor"]

    # Compatible when active excludes match
    assert _is_calibration_mode_compatible(calib, excludes=["vendor", "tests"])
    assert _is_calibration_mode_compatible(calib, excludes=["tests", "vendor"])

    # Incompatible when active excludes differ or are empty
    assert not _is_calibration_mode_compatible(calib, excludes=None)
    assert not _is_calibration_mode_compatible(calib, excludes=["vendor"])
    assert not _is_calibration_mode_compatible(calib, excludes=["other"])

def test_calibration_scope_compatibility_and_rejection() -> None:
    """Verifies that corpus calibration scope is validated and rejected when target scopes differ."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    # Base calibration recorded for "src"
    calib_src = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
        "scope": "src",
        "target_repo_relative": "src",
    }
    calib_src["config_hash"] = compute_calibration_config_hash(calib_src)

    # Active scan on root repo (target_scope=None) must be rejected
    assert not _is_calibration_mode_compatible(calib_src, target_scope=None)
    assert not _is_calibration_mode_compatible(calib_src, target_scope="")

    # Active scan on "tests" must be rejected
    assert not _is_calibration_mode_compatible(calib_src, target_scope="tests")

    # Active scan on "src" matches
    assert _is_calibration_mode_compatible(calib_src, target_scope="src")

    # Base calibration recorded for root repo (scope=None)
    calib_root = {
        "total_units": 500,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
        "scope": None,
    }
    calib_root["config_hash"] = compute_calibration_config_hash(calib_root)

    # Active scan on "src" must be rejected
    assert not _is_calibration_mode_compatible(calib_root, target_scope="src")

    # Active scan on root repo matches
    assert _is_calibration_mode_compatible(calib_root, target_scope=None)
    assert _is_calibration_mode_compatible(calib_root, target_scope="")

    # Config hashes differ between scopes
    hash_src = compute_calibration_config_hash({"scope": "src"})
    hash_root = compute_calibration_config_hash({"scope": None})
    assert hash_src != hash_root

def test_calibration_persisted_fields_validated_against_config_hash() -> None:
    """Verifies that _is_calibration_mode_compatible does not accept calibrations whose stored fields differ from config_hash."""
    from typing import Dict, Any  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    valid_calib: Dict[str, Any] = {
        "total_units": 50,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
        "bag_of_tokens": False,
        "call_sequences": False,
        "filter_stop_shingles": False,
        "scope": "src",
    }
    valid_calib["config_hash"] = compute_calibration_config_hash(valid_calib)

    # 1. Matches when persisted fields match config_hash and active scan matches
    assert _is_calibration_mode_compatible(valid_calib, bag_of_tokens=False, target_scope="src")

    # 2. Desynchronized field: config_hash was for bag_of_tokens=False, but stored field was edited to True
    desync_calib = dict(valid_calib)
    desync_calib["bag_of_tokens"] = True
    # Fast-path must NOT accept desync_calib even if active scan has bag_of_tokens=False (matching config_hash)
    assert not _is_calibration_mode_compatible(desync_calib, bag_of_tokens=False, target_scope="src")

    # 3. Desynchronized scope: config_hash was for scope="src", but stored field was edited to "tests"
    desync_scope = dict(valid_calib)
    desync_scope["scope"] = "tests"
    assert not _is_calibration_mode_compatible(desync_scope, bag_of_tokens=False, target_scope="src")

def test_record_baseline_finalizes_calibration_scope_and_hash(tmp_path: Path) -> None:
    """Verifies record_baseline applies final target scope before hashing and recomputes config_hash."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_calibration_config_hash,
        compute_corpus_calibration,
    )
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    f1 = src / "a.py"
    f2 = src / "b.py"
    f1.write_text("def run():\n    pass\n", encoding="utf-8")
    f2.write_text("def run():\n    pass\n", encoding="utf-8")

    u1 = {"file": "a.py", "name": "run", "shingles": ["run"]}
    u2 = {"file": "b.py", "name": "run", "shingles": ["run"]}
    clones = [(1.0, u1, u2)]

    # 1. Create corpus calibration without explicit scope (scope defaults to None)
    calib_raw = compute_corpus_calibration([u1, u2])
    assert calib_raw.get("scope") is None
    initial_hash = calib_raw.get("config_hash")
    assert initial_hash is not None

    # 2. Record baseline for subdirectory src/ within repo
    base_file = tmp_path / "baseline_calib.json"
    record_baseline(
        clones,
        str(base_file),
        target=str(src),
        repo_root=str(repo),
        threshold=0.9,
        corpus_calibration=calib_raw,
    )

    with open(base_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Scope in calibration entry must be finalized to "src" and not overwritten to None
    persisted_calib = data["corpus_calibration"]
    assert persisted_calib["scope"] == "src"
    assert persisted_calib["target_repo_relative"] == "src"

    # config_hash must be recomputed from the finalized entry (different from initial_hash with scope=None)
    final_expected_hash = compute_calibration_config_hash(persisted_calib)
    assert persisted_calib["config_hash"] == final_expected_hash
    assert data["config_hash"] == final_expected_hash
    assert persisted_calib["config_hash"] != initial_hash

    # 3. Reload baseline and verify compatibility
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["scope"] == "src"
    assert loaded.corpus_calibration["config_hash"] == final_expected_hash

    # Must be compatible with active subdirectory scan (target_scope="src")
    assert _is_calibration_mode_compatible(loaded.corpus_calibration, target_scope="src")
    # Must reject incompatible target_scope (e.g. root scan target_scope=None)
    assert not _is_calibration_mode_compatible(loaded.corpus_calibration, target_scope=None)

def test_load_baseline_preserves_empty_shingle_frequencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that load_baseline preserves an empty shingle_frequencies dictionary as {} rather than None."""
    from pydoppelgangerhunt import matcher  # pylint: disable=import-outside-toplevel

    base_file_empty = tmp_path / "baseline_empty_freqs.json"
    raw_data_empty: Dict[str, Any] = {
        "version": "1.5.0",
        "clone_count": 0,
        "fingerprints": [],
        "corpus_calibration": {
            "total_units": 0,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [],
            "shingle_frequencies": {},
            "min_lines": 4,
            "min_tokens": 10,
        },
    }
    base_file_empty.write_text(json.dumps(raw_data_empty), encoding="utf-8")

    loaded_empty = load_baseline(str(base_file_empty))
    assert loaded_empty.corpus_calibration is not None
    assert isinstance(loaded_empty.corpus_calibration["shingle_frequencies"], dict)
    assert loaded_empty.corpus_calibration["shingle_frequencies"] == {}

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    code1 = (
        "def sample_fn_one(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n"
    )
    code2 = (
        "def sample_fn_two(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n"
    )
    (src_dir / "mod1.py").write_text(code1, encoding="utf-8")
    (src_dir / "mod2.py").write_text(code2, encoding="utf-8")

    # In modern empty calibration, novel shingles are budget-bounded
    monkeypatch.setattr(matcher, "MAX_NOVEL_SHINGLE_PAIR_BUDGET", 0)
    clones_bounded = scan_target(
        str(src_dir),
        min_lines=4,
        min_tokens=10,
        threshold=0.80,
        corpus_calibration=loaded_empty.corpus_calibration,
    )
    # Budget of 0 rejects novel candidate pairs
    assert len(clones_bounded) == 0

    # Contrast with legacy baseline where shingle_frequencies is omitted (None)
    base_file_legacy = tmp_path / "baseline_legacy.json"
    raw_data_legacy: Dict[str, Any] = {
        "version": "1.4.0",
        "clone_count": 0,
        "fingerprints": [],
        "corpus_calibration": {
            "total_units": 10,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [],
            "min_lines": 4,
            "min_tokens": 10,
        },
    }
    base_file_legacy.write_text(json.dumps(raw_data_legacy), encoding="utf-8")
    loaded_legacy = load_baseline(str(base_file_legacy))
    assert loaded_legacy.corpus_calibration is not None
    assert loaded_legacy.corpus_calibration["shingle_frequencies"] is None

    # In legacy calibration, novel budget capping is bypassed
    clones_legacy = scan_target(
        str(src_dir),
        min_lines=4,
        min_tokens=10,
        threshold=0.80,
        corpus_calibration=loaded_legacy.corpus_calibration,
    )
    assert len(clones_legacy) == 1

def test_compute_corpus_calibration_min_frequency_cutoff() -> None:
    """Verifies that compute_corpus_calibration prunes shingles below min_frequency."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel

    u1 = {"file": "a.py", "name": "f1", "shingles": ["singleton_a", "shared"]}
    u2 = {"file": "b.py", "name": "f2", "shingles": ["singleton_b", "shared"]}
    u3 = {"file": "c.py", "name": "f3", "shingles": ["shared"]}

    # Default min_frequency=1 keeps all shingles
    calib_all = compute_corpus_calibration([u1, u2, u3])
    freqs_all = calib_all["shingle_frequencies"]
    assert "singleton_a" in freqs_all
    assert freqs_all["singleton_a"] == 1
    assert "singleton_b" in freqs_all
    assert freqs_all["singleton_b"] == 1
    assert "shared" in freqs_all
    assert freqs_all["shared"] == 3

    # Cutoff min_frequency=2 prunes singletons (df < 2)
    calib_pruned = compute_corpus_calibration([u1, u2, u3], min_frequency=2)
    freqs_pruned = calib_pruned["shingle_frequencies"]
    assert "singleton_a" not in freqs_pruned
    assert "singleton_b" not in freqs_pruned
    assert "shared" in freqs_pruned
    assert freqs_pruned["shared"] == 3

def test_compute_calibration_config_hash_min_frequency() -> None:
    """Verifies that compute_calibration_config_hash treats min_frequency=1 backward-compatibly and hashes >1 thresholds."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel

    # Default / legacy (unspecified or min_frequency=1) produce identical hash
    hash_empty = compute_calibration_config_hash({})
    hash_default = compute_calibration_config_hash({"min_frequency": 1})
    assert hash_empty == hash_default

    # min_frequency > 1 produces distinct hash
    hash_freq2 = compute_calibration_config_hash({"min_frequency": 2})
    assert hash_freq2 != hash_default

    # Alias min_calibration_frequency produces identical hash to min_frequency
    hash_alias = compute_calibration_config_hash({"min_calibration_frequency": 2})
    assert hash_alias == hash_freq2

def test_is_calibration_mode_compatible_min_frequency_fallback() -> None:
    """Verifies that _is_calibration_mode_compatible rejects mismatched min_frequency on fallback path."""
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    # Unhashed or legacy calibration with min_frequency=2
    calib_freq2 = {
        "bag_of_tokens": False,
        "call_sequences": False,
        "filter_stop_shingles": False,
        "audit_tests": False,
        "include_notebooks": False,
        "max_index_frequency": 0.25,
        "min_frequency": 2,
    }

    # Compatible when active scan also specifies min_frequency=2
    assert _is_calibration_mode_compatible(calib_freq2, min_frequency=2) is True

    # Incompatible when active scan runs with default min_frequency=1
    assert _is_calibration_mode_compatible(calib_freq2, min_frequency=1) is False

    # Unhashed calibration with default min_frequency=1
    calib_freq1 = {
        "bag_of_tokens": False,
        "call_sequences": False,
        "filter_stop_shingles": False,
        "audit_tests": False,
        "include_notebooks": False,
        "max_index_frequency": 0.25,
        "min_frequency": 1,
    }
    assert _is_calibration_mode_compatible(calib_freq1, min_frequency=1) is True
    assert _is_calibration_mode_compatible(calib_freq1, min_frequency=2) is False

def test_compute_corpus_calibration_min_calibration_frequency_alias() -> None:
    """Verifies that compute_corpus_calibration resolves min_calibration_frequency keyword alias."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel

    u1 = {"name": "u1", "shingles": ["shared_shingle", "singleton_1"]}
    u2 = {"name": "u2", "shingles": ["shared_shingle", "singleton_2"]}

    # Pass min_calibration_frequency=2 via kwargs (with default min_frequency=1)
    calib = compute_corpus_calibration([u1, u2], min_calibration_frequency=2)

    assert calib["min_frequency"] == 2
    # Singletons must be pruned when min_calibration_frequency=2 is passed
    freqs = calib["shingle_frequencies"]
    assert "shared_shingle" in freqs
    assert freqs["shared_shingle"] == 2
    assert "singleton_1" not in freqs
    assert "singleton_2" not in freqs

def test_compute_corpus_calibration_reduction_metrics_calculation() -> None:
    """Verifies that compute_corpus_calibration calculates shingle reduction counts and compression ratios."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
    )

    # 4 units: 2 shared shingles across all units, 4 singleton shingles (1 per unit)
    u1 = {"name": "u1", "shingles": ["sh_shared_1", "sh_shared_2", "sh_single_1"]}
    u2 = {"name": "u2", "shingles": ["sh_shared_1", "sh_shared_2", "sh_single_2"]}
    u3 = {"name": "u3", "shingles": ["sh_shared_1", "sh_shared_2", "sh_single_3"]}
    u4 = {"name": "u4", "shingles": ["sh_shared_1", "sh_shared_2", "sh_single_4"]}
    units = [u1, u2, u3, u4]

    # Baseline with default min_frequency=1 (no pruning)
    calib_mf1 = compute_corpus_calibration(units, min_frequency=1)
    assert calib_mf1["raw_shingle_count"] == 6
    assert calib_mf1["retained_shingle_count"] == 6
    assert calib_mf1["pruned_shingle_count"] == 0
    assert calib_mf1["shingle_reduction_ratio"] == 0.0
    assert calib_mf1["shingle_compression_ratio"] == 1.0
    assert calib_mf1["bytes_saved_estimate"] == 0
    assert calib_mf1["bytes_reduction_ratio"] == 0.0
    assert calib_mf1["bytes_compression_ratio"] == 1.0
    assert isinstance(calib_mf1["reduction_stats"], dict)

    # Baseline with min_frequency=2 (prunes all 4 singletons, retains 2 shared shingles)
    calib_mf2 = compute_corpus_calibration(units, min_frequency=2)
    assert calib_mf2["raw_shingle_count"] == 6
    assert calib_mf2["retained_shingle_count"] == 2
    assert calib_mf2["pruned_shingle_count"] == 4
    assert calib_mf2["shingle_reduction_ratio"] == round(4 / 6, 4)
    assert calib_mf2["shingle_compression_ratio"] == 3.0  # 6 / 2
    assert calib_mf2["bytes_saved_estimate"] > 0
    assert calib_mf2["bytes_reduction_ratio"] > 0.0
    assert calib_mf2["bytes_compression_ratio"] > 1.0

    # Test reduction_stats dictionary sub-object matches top-level metrics
    stats = calib_mf2["reduction_stats"]
    assert stats["raw_shingle_count"] == 6
    assert stats["retained_shingle_count"] == 2
    assert stats["pruned_shingle_count"] == 4
    assert stats["shingle_reduction_ratio"] == round(4 / 6, 4)
    assert stats["shingle_compression_ratio"] == 3.0

def test_corpus_calibration_reduction_metrics_roundtrip_and_config_hash_stability(tmp_path: Path) -> None:
    """Verifies that record_baseline and load_baseline preserve reduction metrics without altering config_hash."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_calibration_config_hash,
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )

    u1 = {"name": "u1", "shingles": ["common", "u1_only"]}
    u2 = {"name": "u2", "shingles": ["common", "u2_only"]}
    calib = compute_corpus_calibration([u1, u2], min_frequency=2)

    expected_hash = compute_calibration_config_hash(calib)
    assert calib["config_hash"] == expected_hash

    baseline_file = tmp_path / "baseline_calib_reduction.json"
    record_baseline(
        [],
        str(baseline_file),
        target=str(tmp_path),
        threshold=0.85,
        corpus_calibration=calib,
    )

    loaded = load_baseline(str(baseline_file))
    assert loaded.corpus_calibration is not None
    assert loaded.config_hash == expected_hash
    assert loaded.corpus_calibration["config_hash"] == expected_hash

    # Verify reduction metrics round-tripped accurately
    assert loaded.corpus_calibration["raw_shingle_count"] == 3
    assert loaded.corpus_calibration["retained_shingle_count"] == 1
    assert loaded.corpus_calibration["pruned_shingle_count"] == 2
    assert loaded.corpus_calibration["shingle_reduction_ratio"] == round(2 / 3, 4)
    assert loaded.corpus_calibration["shingle_compression_ratio"] == 3.0
    assert loaded.calibration_reduction_ratio == round(2 / 3, 4)
    assert loaded.calibration_compression_ratio == 3.0
    assert "reduction_stats" in loaded.corpus_calibration

def test_corpus_calibration_reduction_metrics_empty_units_and_edge_cases() -> None:
    """Verifies boundary conditions for reduction metric calculations (empty units, all pruned)."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
    )

    # Empty corpus
    calib_empty = compute_corpus_calibration([], min_frequency=2)
    assert calib_empty["raw_shingle_count"] == 0
    assert calib_empty["retained_shingle_count"] == 0
    assert calib_empty["pruned_shingle_count"] == 0
    assert calib_empty["shingle_reduction_ratio"] == 0.0
    assert calib_empty["shingle_compression_ratio"] == 1.0

    # Corpus where all shingles are singletons and pruned
    u_single = [{"name": "singleton_only", "shingles": ["unique_shingle_xyz"]}]
    calib_all_pruned = compute_corpus_calibration(u_single, min_frequency=2)
    assert calib_all_pruned["raw_shingle_count"] == 1
    assert calib_all_pruned["retained_shingle_count"] == 0
    assert calib_all_pruned["pruned_shingle_count"] == 1
    assert calib_all_pruned["shingle_reduction_ratio"] == 1.0
    assert calib_all_pruned["shingle_compression_ratio"] == 1.0
