"""Pure graph algorithms on plain dict adjacency (no Gramps/gi dependency).

Kept dependency-free so unit tests run anywhere. The DB adapter in
graph_analysis.py builds the adjacency and calls into here.
"""
from __future__ import annotations

from collections import defaultdict, deque
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


def betweenness_centrality(adj: Adjacency) -> list[tuple[Hashable, float]]:
    """Unweighted betweenness (Brandes). Undirected → divide final scores by 2."""
    bc: dict[Hashable, float] = {n: 0.0 for n in adj}
    for s in adj:
        stack: list[Hashable] = []
        preds: dict[Hashable, list[Hashable]] = {n: [] for n in adj}
        sigma: dict[Hashable, float] = {n: 0.0 for n in adj}
        dist: dict[Hashable, int] = {n: -1 for n in adj}
        sigma[s] = 1.0
        dist[s] = 0
        queue: deque[Hashable] = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj.get(v, ()):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta: dict[Hashable, float] = defaultdict(float)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return sorted(
        ((n, v / 2.0) for n, v in bc.items()),
        key=lambda kv: (-kv[1], str(kv[0])),
    )


def closeness_centrality(adj: Adjacency) -> list[tuple[Hashable, float]]:
    """Component-local closeness: (reachable-1) / sum(distances)."""
    out: list[tuple[Hashable, float]] = []
    for s in adj:
        dist: dict[Hashable, int] = {s: 0}
        queue: deque[Hashable] = deque([s])
        while queue:
            v = queue.popleft()
            for w in adj.get(v, ()):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        total = sum(dist.values())
        score = ((len(dist) - 1) / total) if total > 0 else 0.0
        out.append((s, score))
    return sorted(out, key=lambda kv: (-kv[1], str(kv[0])))


def ancestry_depth(parents: dict[Hashable, list[Hashable]]) -> dict[Hashable, int]:
    """Longest number of generations UP from each node (0 = no known parents).

    *parents* maps node -> list of parent nodes. Assumes a DAG; a defensive
    visiting-set breaks any accidental cycle (returns partial depth, no crash).
    """
    memo: dict[Hashable, int] = {}

    def depth(n: Hashable, visiting: set[Hashable]) -> int:
        if n in memo:
            return memo[n]
        if n in visiting:            # cycle guard
            return 0
        ps = parents.get(n, ())
        if not ps:
            memo[n] = 0
            return 0
        visiting.add(n)
        best = 1 + max(depth(p, visiting) for p in ps)
        visiting.discard(n)
        memo[n] = best
        return best

    for node in parents:
        depth(node, set())
    return memo


def roots(parents: dict[Hashable, list[Hashable]]) -> set[Hashable]:
    """Nodes with no recorded parents (genealogical 'brick walls')."""
    return {n for n in parents if not parents.get(n)}
