"""Disjoint-Set Union clustering and Clone Family aggregation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


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


def unit_key(unit: Dict[str, Any]) -> str:
    """Generates unique deterministic string key for an AST unit."""
    norm_file = unit["file"].replace("\\", "/")
    return f"{norm_file}:{unit['start']}-{unit['end']}:{unit['name']}"


def compute_medoid(
    member_keys: List[str],
    sim_matrix: Dict[Tuple[str, str], float],
) -> Tuple[str, float]:
    """Finds the representative medoid unit maximizing total similarity to other members.

    Returns:
        A tuple of (medoid_key, coherence_score), where coherence_score is the
        mean pairwise similarity of the medoid to all members in the cluster.
    """
    if not member_keys:
        raise ValueError("Cannot compute medoid of an empty member set.")
    if len(member_keys) == 1:
        return member_keys[0], 1.0

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

    return best_key, best_score


def cluster_clone_families(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    linkage: str = "single",
    min_similarity_floor: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Groups pairwise clone detections into Clone Families with configurable linkage.

    Supported linkage strategies:
    - 'single': Connected components via Union-Find (transitive chaining).
    - 'complete': Clique partitioning where all pairwise edges must exist and satisfy min_similarity_floor.
    - 'average': Hierarchical agglomerative clustering requiring mean cross-cluster similarity >= min_similarity_floor.
    - 'medoid': Centroid-based agglomerative clustering tracking a dynamic medoid and requiring similarity to medoid >= min_similarity_floor.
    """
    if not clones:
        return []

    unit_map: Dict[str, Dict[str, Any]] = {}
    sim_matrix: Dict[Tuple[str, str], float] = {}
    pair_records: List[Tuple[float, str, str]] = []

    for sim, u1, u2 in clones:
        k1 = unit_key(u1)
        k2 = unit_key(u2)
        unit_map[k1] = u1
        unit_map[k2] = u2
        sim_matrix[(k1, k2)] = sim
        sim_matrix[(k2, k1)] = sim
        pair_records.append((sim, k1, k2))

    all_keys = sorted(unit_map.keys())

    floor = (
        min_similarity_floor
        if min_similarity_floor is not None
        else (min(sim for sim, _, _ in clones) if clones else 0.0)
    )

    raw_clusters: List[List[str]] = []

    if linkage == "single":
        uf = UnionFind()
        for _, k1, k2 in pair_records:
            uf.union(k1, k2)

        groups: Dict[str, List[str]] = {}
        for k in all_keys:
            root = uf.find(k)
            groups.setdefault(root, []).append(k)
        raw_clusters = list(groups.values())

    elif linkage == "complete":
        sorted_pairs = sorted(pair_records, key=lambda x: (x[0], x[1], x[2]), reverse=True)
        cluster_map: Dict[str, Set[str]] = {k: {k} for k in all_keys}

        for _, k1, k2 in sorted_pairs:
            c1 = cluster_map[k1]
            c2 = cluster_map[k2]
            if c1 is c2:
                continue

            can_merge = True
            for u in c1:
                for v in c2:
                    if sim_matrix.get((u, v), 0.0) < floor:
                        can_merge = False
                        break
                if not can_merge:
                    break

            if can_merge:
                merged = c1 | c2
                for node in merged:
                    cluster_map[node] = merged

        unique_clusters: List[Set[str]] = []
        seen_ids: Set[int] = set()
        for c in cluster_map.values():
            cid = id(c)
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_clusters.append(c)
        raw_clusters = [sorted(c) for c in unique_clusters]

    elif linkage == "average":
        sorted_pairs = sorted(pair_records, key=lambda x: (x[0], x[1], x[2]), reverse=True)
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

        unique_clusters = []
        seen_ids = set()
        for c in cluster_map.values():
            cid = id(c)
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_clusters.append(c)
        raw_clusters = [sorted(c) for c in unique_clusters]

    elif linkage in ("medoid", "centroid"):
        sorted_pairs = sorted(pair_records, key=lambda x: (x[0], x[1], x[2]), reverse=True)
        cluster_map = {k: {k} for k in all_keys}

        for _, k1, k2 in sorted_pairs:
            c1 = cluster_map[k1]
            c2 = cluster_map[k2]
            if c1 is c2:
                continue

            candidate_merged = sorted(c1 | c2)
            cand_medoid, _ = compute_medoid(candidate_merged, sim_matrix)

            can_merge = all(
                (node == cand_medoid or sim_matrix.get((node, cand_medoid), 0.0) >= floor)
                for node in candidate_merged
            )

            if can_merge:
                merged = set(candidate_merged)
                for node in merged:
                    cluster_map[node] = merged

        unique_clusters = []
        seen_ids = set()
        for c in cluster_map.values():
            cid = id(c)
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_clusters.append(c)
        raw_clusters = [sorted(c) for c in unique_clusters]

    else:
        raise ValueError(
            f"Unsupported linkage strategy '{linkage}'. Expected 'single', 'complete', 'average', or 'medoid'."
        )

    clusters = [m for m in raw_clusters if len(m) >= 2]

    families: List[Dict[str, Any]] = []
    for idx, member_keys in enumerate(clusters):
        members = [unit_map[k] for k in member_keys]
        members.sort(key=lambda u: (u["file"].replace("\\", "/"), u["start"]))
        member_set = set(member_keys)

        family_sims = [
            sim for sim, k1, k2 in pair_records if k1 in member_set and k2 in member_set
        ]
        unique_files = sorted(list({u["file"].replace("\\", "/") for u in members}))
        total_lines = sum(u["end"] - u["start"] + 1 for u in members)
        avg_sim = (sum(family_sims) / len(family_sims)) if family_sims else 1.0
        max_sim = max(family_sims) if family_sims else 1.0
        min_sim = min(family_sims) if family_sims else 1.0

        medoid_key, coherence = compute_medoid(member_keys, sim_matrix)
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

    families.sort(key=lambda f: (f["member_count"], f["avg_similarity"]), reverse=True)
    for idx, fam in enumerate(families):
        fam["family_id"] = f"CF-{idx + 1:03d}"

    return families
