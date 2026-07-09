"""Pure unit tests for graph_primitives (no gi required)."""
from gramps_webapi.api.resources.graph_primitives import connected_components


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
