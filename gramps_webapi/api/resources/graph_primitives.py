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


def articulation_points(adj: Adjacency) -> set[Hashable]:
    """Return the set of articulation points (cut vertices) of an undirected graph.

    Iterative Tarjan lowlink over each component. A node whose removal increases
    the component count. Handles forests (multiple roots).
    """
    disc: dict[Hashable, int] = {}
    low: dict[Hashable, int] = {}
    parent: dict[Hashable, Hashable | None] = {}
    ap: set[Hashable] = set()
    timer = 0

    for root in adj:
        if root in disc:
            continue
        # Iterative DFS. Stack holds (node, iterator over neighbours).
        parent[root] = None
        stack: list[tuple[Hashable, "iter"]] = [(root, iter(adj.get(root, ())))]
        disc[root] = low[root] = timer
        timer += 1
        root_children = 0
        while stack:
            node, it = stack[-1]
            advanced = False
            for nb in it:
                if nb not in disc:
                    if node == root:
                        root_children += 1
                    parent[nb] = node
                    disc[nb] = low[nb] = timer
                    timer += 1
                    stack.append((nb, iter(adj.get(nb, ()))))
                    advanced = True
                    break
                elif nb != parent.get(node):
                    low[node] = min(low[node], disc[nb])
            if not advanced:
                stack.pop()
                if stack:
                    par = stack[-1][0]
                    low[par] = min(low[par], low[node])
                    # non-root articulation condition
                    if parent.get(par) is not None and low[node] >= disc[par]:
                        ap.add(par)
        if root_children > 1:
            ap.add(root)
    return ap
