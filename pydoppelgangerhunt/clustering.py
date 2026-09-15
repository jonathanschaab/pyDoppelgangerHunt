"""Disjoint-Set Union clustering and Clone Family aggregation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


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


def cluster_clone_families(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Groups pairwise clone detections into connected component Clone Families using Union-Find."""
    uf = UnionFind()
    unit_map: Dict[str, Dict[str, Any]] = {}
    pair_records: List[Tuple[float, str, str]] = []

    for sim, u1, u2 in clones:
        k1 = unit_key(u1)
        k2 = unit_key(u2)
        unit_map[k1] = u1
        unit_map[k2] = u2
        uf.union(k1, k2)
        pair_records.append((sim, k1, k2))

    family_groups: Dict[str, List[str]] = {}
    for key in unit_map:
        root = uf.find(key)
        family_groups.setdefault(root, []).append(key)

    families: List[Dict[str, Any]] = []
    for idx, (_root, member_keys) in enumerate(family_groups.items()):
        members = [unit_map[k] for k in member_keys]
        members.sort(key=lambda u: (u["file"].replace("\\", "/"), u["start"]))
        member_set = set(member_keys)
        family_sims = [sim for sim, k1, k2 in pair_records if k1 in member_set and k2 in member_set]
        unique_files = sorted(list({u["file"].replace("\\", "/") for u in members}))
        total_lines = sum(u["end"] - u["start"] + 1 for u in members)
        avg_sim = (sum(family_sims) / len(family_sims)) if family_sims else 1.0
        max_sim = max(family_sims) if family_sims else 1.0

        families.append({
            "family_id": f"CF-{idx + 1:03d}",
            "members": members,
            "member_count": len(members),
            "unique_files": unique_files,
            "avg_similarity": avg_sim,
            "max_similarity": max_sim,
            "total_lines": total_lines,
        })

    families.sort(key=lambda f: (f["member_count"], f["avg_similarity"]), reverse=True)
    return families
