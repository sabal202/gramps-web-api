"""Pure unit tests for graph_primitives (no gi required)."""
from gramps_webapi.api.resources.graph_primitives import (
    ancestry_depth,
    articulation_points,
    betweenness_centrality,
    closeness_centrality,
    connected_components,
    degree_centrality,
    roots,
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


def test_articulation_points_finds_cut_vertex():
    # a-b-c-d chain: b and c are cut vertices; a,d are leaves
    adj = {"a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c"}}
    aps = articulation_points(adj)
    assert aps == {"b", "c"}


def test_articulation_points_none_in_cycle():
    adj = {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}}
    assert articulation_points(adj) == set()


def test_betweenness_path_midpoint_highest():
    adj = {"a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c"}}
    bc = dict(betweenness_centrality(adj))
    assert bc["b"] > bc["a"]
    assert bc["c"] > bc["d"]


def test_closeness_center_highest():
    adj = {"h": {"a", "b", "c"}, "a": {"h"}, "b": {"h"}, "c": {"h"}}
    cc = dict(closeness_centrality(adj))
    assert cc["h"] == max(cc.values())


def test_ancestry_depth_counts_longest_chain():
    # child c -> parent b -> parent a ; also c -> parent x (shallow)
    parents = {"c": ["b", "x"], "b": ["a"], "a": [], "x": [], "d": []}
    depth = ancestry_depth(parents)     # {node: longest #generations up}
    assert depth["c"] == 2
    assert depth["b"] == 1
    assert depth["a"] == 0
    assert depth["d"] == 0


def test_roots_are_parentless():
    parents = {"c": ["b"], "b": [], "z": []}
    assert roots(parents) == {"b", "z"}
