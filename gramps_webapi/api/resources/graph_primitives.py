"""Pure graph algorithms on plain dict adjacency (no Gramps/gi dependency).

Kept dependency-free so unit tests run anywhere. The DB adapter in
graph_analysis.py builds the adjacency and calls into here.
"""
from __future__ import annotations

from collections import deque
from typing import Hashable

Adjacency = dict[Hashable, set[Hashable]]


def connected_components(adj: Adjacency) -> list[set[Hashable]]:
    """Return connected components of an undirected graph, largest first.

    Every key in *adj* is a node (isolated nodes have an empty neighbour set),
    so singletons appear as their own component.
    """
    seen: set[Hashable] = set()
    components: list[set[Hashable]] = []
    for start in adj:
        if start in seen:
            continue
        comp: set[Hashable] = set()
        queue: deque[Hashable] = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            comp.add(node)
            for nb in adj.get(node, ()):  # tolerate one-sided edges
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        components.append(comp)
    components.sort(key=len, reverse=True)
    return components


def degree_centrality(adj: Adjacency) -> list[tuple[Hashable, int]]:
    """Nodes ranked by neighbour count, descending (ties: node order stable)."""
    return sorted(
        ((n, len(adj.get(n, ()))) for n in adj),
        key=lambda kv: (-kv[1], str(kv[0])),
    )
