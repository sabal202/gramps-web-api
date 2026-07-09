"""Pure unit tests for graph_primitives (no gi required)."""
from gramps_webapi.api.resources.graph_primitives import (
    connected_components,
    degree_centrality,
)


def test_connected_components_splits_islands():
    adj = {
        "a": {"b"}, "b": {"a"},          # component 1 (pair)
        "c": {"d"}, "d": {"c", "e"}, "e": {"d"},  # component 2 (triple)
        "f": set(),                       # singleton
    }
    comps = connected_components(adj)
    # sorted largest-first, each a set of nodes
    assert [len(c) for c in comps] == [3, 2, 1]
    assert {"c", "d", "e"} in comps
    assert {"f"} in comps


def test_degree_centrality_ranks_hub():
    adj = {"h": {"a", "b", "c"}, "a": {"h"}, "b": {"h"}, "c": {"h"}}
    ranked = degree_centrality(adj)          # list[(node, degree)] desc
    assert ranked[0] == ("h", 3)
    assert {n for n, _ in ranked} == {"h", "a", "b", "c"}
