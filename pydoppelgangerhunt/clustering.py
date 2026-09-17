"""Disjoint-Set Union clustering and Clone Family aggregation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydoppelgangerhunt.config import canonical_path_key


class UnionFind:
    """Disjoint-Set Union (DSU) data structure with path compression and union by rank."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, item: str) -> str:
        """Finds representative set identifier for an item with path compression."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> str:
        """Merges sets containing item1 and item2 using union by rank."""
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 == root2:
            return root1
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
            return root2
        if self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
            return root1
        self.parent[root2] = root1
        self.rank[root1] += 1
        return root1


def _normalize_unit_file(unit: Dict[str, Any]) -> str:
    return canonical_path_key(str(unit.get("file") or ""), strip_anchor=True)


def unit_key(unit: Dict[str, Any]) -> str:
    """Generates unique deterministic string key for an AST unit."""
    norm_file = canonical_path_key(str(unit.get("file") or ""), strip_anchor=False)
    s = int(unit.get("start") or 1)
    e = int(unit.get("end") or s)
    name = str(unit.get("name") or "unit")
    return f"{norm_file}:{s}-{e}:{name}"


def compute_medoid(
    member_keys: Sequence[str],
    sim_matrix: Dict[Tuple[str, str], float],
    cache: Optional[Dict[Tuple[str, ...], Tuple[str, float]]] = None,
) -> Tuple[str, float]:
    """Finds the representative medoid unit maximizing total similarity to other members.

    Args:
        member_keys: Collection of unique unit string keys in the cluster.
        sim_matrix: Pairwise similarity lookup mapping (key1, key2) to similarity score.
        cache: Optional memoization dictionary mapping sorted member key tuples to
            (medoid_key, coherence_score). Memoization avoids redundant O(N^2)
            recalculations across iterative agglomerative clustering passes on large clusters.

    Returns:
        A tuple of (medoid_key, coherence_score), where coherence_score is the
        mean pairwise similarity of the medoid to all members in the cluster.
    """
    if not member_keys:
        raise ValueError("Cannot compute medoid of an empty member set.")
    if len(member_keys) == 1:
        return member_keys[0], 1.0

    cache_key: Optional[Tuple[str, ...]] = None
    if cache is not None:
        cache_key = tuple(sorted(member_keys))
        if cache_key in cache:
            return cache[cache_key]

    best_key = member_keys[0]
    best_score = -1.0

    for u in member_keys:
        total_sim = 0.0
        for v in member_keys:
            if u == v:
                total_sim += 1.0
            else:
                total_sim += sim_matrix.get((u, v), sim_matrix.get((v, u), 0.0))
        mean_sim = total_sim / len(member_keys)
        if mean_sim > best_score or (mean_sim == best_score and u < best_key):
            best_score = mean_sim
            best_key = u

    res = (best_key, best_score)
    if cache is not None and cache_key is not None:
        cache[cache_key] = res
    return res


def cluster_clone_families(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    linkage: str = "single",
    min_similarity_floor: Optional[float] = None,
    linkage_tolerance: float = 0.0,
) -> List[Dict[str, Any]]:
    """Groups pairwise clone detections into Clone Families with configurable linkage.

    Supported linkage strategies:
    - 'single': Connected components via Union-Find (transitive chaining).
    - 'complete': Clique partitioning where all pairwise edges must exist and satisfy min_similarity_floor.
      When linkage_tolerance > 0.0, allows edges >= (min_similarity_floor - linkage_tolerance)
      provided mean cross-cluster similarity >= min_similarity_floor.
    - 'quasi_complete': Tolerance-bounded complete linkage designed for Type-3 clone families,
      defaulting to linkage_tolerance=0.05 if unspecified. Requires all cross-cluster edges
      to meet >= (min_similarity_floor - linkage_tolerance) and mean cross-cluster similarity
      >= min_similarity_floor.
    - 'average': Hierarchical agglomerative clustering requiring mean cross-cluster pairwise
      similarity >= min_similarity_floor. Note that this enforces "average threshold" semantics
      over all cross-cluster candidate pairs, unlike complete-linkage's strict "all pairs"
      requirement. Pairs between clusters that do not share an edge in 'clones' contribute 0.0
      similarity to the mean cross-cluster evaluation.
    - 'medoid': Centroid-based agglomerative clustering tracking a dynamic medoid and requiring
      similarity to medoid >= min_similarity_floor.
    """
    if not clones:
        return []

    unit_map: Dict[str, Dict[str, Any]] = {}
    sim_matrix: Dict[Tuple[str, str], float] = {}
    deduped_pairs: Dict[Tuple[str, str], float] = {}

    for sim, u1, u2 in clones:
        k1 = unit_key(u1)
        k2 = unit_key(u2)
        unit_map[k1] = u1
        unit_map[k2] = u2
        sim_matrix[(k1, k2)] = max(sim_matrix.get((k1, k2), 0.0), sim)
        sim_matrix[(k2, k1)] = max(sim_matrix.get((k2, k1), 0.0), sim)

        edge_key = (k1, k2) if k1 <= k2 else (k2, k1)
        deduped_pairs[edge_key] = max(deduped_pairs.get(edge_key, 0.0), sim)

    all_keys = sorted(unit_map.keys())

    # Sorted candidate pairs:
    # 1. Higher similarity first (-round(sim, 9))
    # 2. Ascending tie-breaking by source key (k1)
    # 3. Ascending tie-breaking by target key (k2)
    sorted_pairs: List[Tuple[float, str, str]] = [
        (sim, edge[0], edge[1])
        for edge, sim in sorted(
            deduped_pairs.items(),
            key=lambda item: (-round(item[1], 9), item[0][0], item[0][1]),
        )
    ]

    floor = (
        min_similarity_floor
        if min_similarity_floor is not None
        else (min(sim for sim, _, _ in clones) if clones else 0.0)
    )

    effective_tolerance = linkage_tolerance
    if linkage == "quasi_complete" and effective_tolerance == 0.0:
        effective_tolerance = 0.05

    raw_clusters: List[List[str]] = []
    medoid_cache: Dict[Tuple[str, ...], Tuple[str, float]] = {}

    if linkage == "single":
        uf = UnionFind()
        for _, k1, k2 in sorted_pairs:
            uf.union(k1, k2)

        groups: Dict[str, List[str]] = {}
        for k in all_keys:
            root = uf.find(k)
            groups.setdefault(root, []).append(k)
        unique_groups = [sorted(members) for members in groups.values()]
        raw_clusters = sorted(unique_groups, key=lambda g: (g[0], len(g)))

    elif linkage in ("complete", "quasi_complete"):
        cluster_map: Dict[str, Set[str]] = {k: {k} for k in all_keys}
        min_allowed_edge = max(0.0, floor - effective_tolerance)

        for _, k1, k2 in sorted_pairs:
            c1 = cluster_map[k1]
            c2 = cluster_map[k2]
            if c1 is c2:
                continue

            can_merge = True
            total_cross_sim = 0.0
            for u in c1:
                for v in c2:
                    edge_sim = sim_matrix.get((u, v), 0.0)
                    if edge_sim < min_allowed_edge:
                        can_merge = False
                        break
                    total_cross_sim += edge_sim
                if not can_merge:
                    break

            if can_merge and effective_tolerance > 0.0:
                avg_cross_sim = total_cross_sim / (len(c1) * len(c2))
                if avg_cross_sim < floor:
                    can_merge = False

            if can_merge:
                merged = c1 | c2
                for node in merged:
                    cluster_map[node] = merged

        unique_clusters_dict: Dict[Tuple[str, ...], List[str]] = {}
        for k in all_keys:
            c = cluster_map[k]
            rep = tuple(sorted(c))
            if rep not in unique_clusters_dict:
                unique_clusters_dict[rep] = list(rep)
        raw_clusters = [unique_clusters_dict[rep] for rep in sorted(unique_clusters_dict.keys())]

    elif linkage == "average":
        cluster_map = {k: {k} for k in all_keys}

        for _, k1, k2 in sorted_pairs:
            c1 = cluster_map[k1]
            c2 = cluster_map[k2]
            if c1 is c2:
                continue

            total_cross_sim = sum(sim_matrix.get((u, v), 0.0) for u in c1 for v in c2)
            avg_cross_sim = total_cross_sim / (len(c1) * len(c2))

            if avg_cross_sim >= floor:
                merged = c1 | c2
                for node in merged:
                    cluster_map[node] = merged

        unique_clusters_dict = {}
        for k in all_keys:
            c = cluster_map[k]
            rep = tuple(sorted(c))
            if rep not in unique_clusters_dict:
                unique_clusters_dict[rep] = list(rep)
        raw_clusters = [unique_clusters_dict[rep] for rep in sorted(unique_clusters_dict.keys())]

    elif linkage in ("medoid", "centroid"):
        cluster_map = {k: {k} for k in all_keys}

        for _, k1, k2 in sorted_pairs:
            c1 = cluster_map[k1]
            c2 = cluster_map[k2]
            if c1 is c2:
                continue

            candidate_merged = sorted(c1 | c2)
            cand_medoid, _ = compute_medoid(
                candidate_merged, sim_matrix, cache=medoid_cache
            )

            can_merge = all(
                (node == cand_medoid or sim_matrix.get((node, cand_medoid), 0.0) >= floor)
                for node in candidate_merged
            )

            if can_merge:
                merged = set(candidate_merged)
                for node in merged:
                    cluster_map[node] = merged

        unique_clusters_dict = {}
        for k in all_keys:
            c = cluster_map[k]
            rep = tuple(sorted(c))
            if rep not in unique_clusters_dict:
                unique_clusters_dict[rep] = list(rep)
        raw_clusters = [unique_clusters_dict[rep] for rep in sorted(unique_clusters_dict.keys())]

    else:
        raise ValueError(
            f"Unsupported linkage strategy '{linkage}'. Expected 'single', 'complete', 'quasi_complete', 'average', or 'medoid'."
        )

    clusters = [m for m in raw_clusters if len(m) >= 2]

    families: List[Dict[str, Any]] = []
    for idx, member_keys in enumerate(clusters):
        members = [unit_map[k] for k in member_keys]
        members.sort(
            key=lambda u: (
                canonical_path_key(str(u.get("file") or ""), strip_anchor=False),
                int(u.get("start") or 1),
            )
        )
        member_set = set(member_keys)

        family_sims = [
            sim for sim, k1, k2 in sorted_pairs if k1 in member_set and k2 in member_set
        ]
        unique_files = sorted(list({_normalize_unit_file(u) for u in members}))
        total_lines = sum(
            int(u.get("end") or int(u.get("start") or 1)) - int(u.get("start") or 1) + 1
            for u in members
        )
        avg_sim = (sum(family_sims) / len(family_sims)) if family_sims else 1.0
        max_sim = max(family_sims) if family_sims else 1.0
        min_sim = min(family_sims) if family_sims else 1.0

        medoid_key, coherence = compute_medoid(
            member_keys, sim_matrix, cache=medoid_cache
        )
        medoid_unit = unit_map[medoid_key]

        families.append({
            "family_id": f"CF-{idx + 1:03d}",
            "members": members,
            "member_count": len(members),
            "unique_files": unique_files,
            "avg_similarity": avg_sim,
            "max_similarity": max_sim,
            "min_similarity": min_sim,
            "medoid": medoid_unit,
            "coherence": coherence,
            "total_lines": total_lines,
        })

    families.sort(
        key=lambda f: (
            -f["member_count"],
            -round(f["avg_similarity"], 9),
            _normalize_unit_file(f["members"][0]),
            int(f["members"][0].get("start") or 1),
            str(f["medoid"].get("name") or "") if isinstance(f.get("medoid"), dict) else "",
        )
    )
    for idx, fam in enumerate(families):
        fam["family_id"] = f"CF-{idx + 1:03d}"

    return families
