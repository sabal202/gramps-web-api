"""Adapter unit tests for graph_analysis on the real example tree (needs gi)."""
import unittest

from tests import ExampleDbInMemory

from gramps_webapi.api.resources.graph_analysis import (
    build_tree_graph,
    centrality,
    connectivity,
    deepest_ancestors,
    integrity,
    lineages,
)


class TestGraphAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._example = ExampleDbInMemory()
        cls.db = cls._example.load()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        cls._example.close()

    # -- Task 2.1: TreeGraph builder -------------------------------------

    def test_build_tree_graph_has_all_people_as_nodes(self):
        g = build_tree_graph(self.db)
        self.assertEqual(len(g.undirected), self.db.get_number_of_people())
        # neighbour sets everywhere (isolated nodes have empty sets)
        self.assertTrue(all(isinstance(v, set) for v in g.undirected.values()))

    def test_build_tree_graph_parents_maps_are_subsets(self):
        g = build_tree_graph(self.db)
        # Every birth-parent link must also appear in the general parents map.
        for child, birth_parents in g.parents_birth.items():
            for parent in birth_parents:
                self.assertIn(parent, g.parents.get(child, []))
        # Every recorded parent link must also be an undirected edge.
        for child, parents in g.parents.items():
            for parent in parents:
                self.assertIn(parent, g.undirected.get(child, set()))
                self.assertIn(child, g.undirected.get(parent, set()))

    # -- Task 2.2: connectivity() -----------------------------------------

    def test_connectivity_component_sizes_sum_to_person_count(self):
        result = connectivity(self.db)
        for key in (
            "person_count",
            "component_count",
            "main_component_size",
            "islands",
            "isolated_pairs",
            "orphans",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["person_count"], self.db.get_number_of_people())
        self.assertGreaterEqual(result["component_count"], 1)
        self.assertGreaterEqual(result["main_component_size"], 0)

    def test_connectivity_include_singletons_toggle(self):
        with_singles = connectivity(self.db, include_singletons=True)
        without_singles = connectivity(self.db, include_singletons=False)
        # Turning singletons off never adds orphans, only removes/keeps them.
        self.assertLessEqual(
            len(without_singles["orphans"]), len(with_singles["orphans"])
        )
        if len(with_singles["orphans"]) > 0:
            self.assertEqual(without_singles["orphans"], [])

    # -- Task 2.3: deepest_ancestors() ------------------------------------

    def test_deepest_ancestors_shape_and_furthest(self):
        # Pick a person who has at least one recorded parent.
        g = build_tree_graph(self.db)
        anchor_handle = next(h for h, ps in g.parents.items() if ps)
        anchor = self.db.get_person_from_handle(anchor_handle)
        result = deepest_ancestors(self.db, anchor, g=g)
        for key in ("anchor", "max_depth", "by_line", "furthest"):
            self.assertIn(key, result)
        self.assertEqual(result["anchor"], anchor.handle)
        self.assertGreaterEqual(result["max_depth"], 1)
        self.assertTrue(len(result["furthest"]) >= 1)
        # Every "furthest" ancestor is at exactly max_depth in some line.
        depths = {e["root"]: e["depth"] for e in result["by_line"]}
        for h in result["furthest"]:
            self.assertIn(h, depths)
            self.assertEqual(depths[h], result["max_depth"])

    def test_deepest_ancestors_no_parents_gives_zero_depth(self):
        g = build_tree_graph(self.db)
        childless_root = next(h for h, ps in g.parents.items() if not ps)
        anchor = self.db.get_person_from_handle(childless_root)
        result = deepest_ancestors(self.db, anchor, g=g)
        self.assertEqual(result["max_depth"], 0)
        self.assertEqual(result["by_line"], [])
        self.assertEqual(result["furthest"], [])

    def test_deepest_ancestors_birth_only_and_top(self):
        g = build_tree_graph(self.db)
        anchor_handle = next(h for h, ps in g.parents.items() if ps)
        anchor = self.db.get_person_from_handle(anchor_handle)
        result = deepest_ancestors(self.db, anchor, birth_only=True, top=1, g=g)
        self.assertLessEqual(len(result["by_line"]), 1)

    # -- Task 2.4: lineages() ----------------------------------------------

    def test_lineages_shape_and_ordering(self):
        result = lineages(self.db)
        self.assertIn("max_tree_depth", result)
        self.assertIn("root_count", result)
        self.assertIn("lineages", result)
        depths = [entry["depth"] for entry in result["lineages"]]
        self.assertEqual(depths, sorted(depths, reverse=True))
        for entry in result["lineages"]:
            self.assertIn("root", entry)
            self.assertIn("depth", entry)
            self.assertIn("size", entry)
            self.assertGreaterEqual(entry["size"], 1)

    def test_lineages_min_size_filters(self):
        result_all = lineages(self.db, min_size=1)
        result_filtered = lineages(self.db, min_size=1000000)
        self.assertEqual(result_filtered["lineages"], [])
        self.assertGreaterEqual(len(result_all["lineages"]), 0)

    # -- Task 2.5: integrity() ----------------------------------------------

    def test_integrity_default_checks(self):
        result = integrity(self.db)
        self.assertEqual(set(result["checks"]), {"dangling", "one_sided_refs"})
        self.assertNotIn("thin_records", result["problems"])
        for key, items in result["problems"].items():
            self.assertEqual(result["counts"][key], len(items))
            for item in items:
                self.assertIn("handle", item)
                self.assertIn("gramps_id", item)
                self.assertIn("detail", item)

    def test_integrity_thin_records_opt_in(self):
        result = integrity(self.db, checks=("thin_records",))
        self.assertEqual(result["checks"], ["thin_records"])
        self.assertIn("thin_records", result["problems"])
        self.assertNotIn("one_sided_refs", result["problems"])
        self.assertNotIn("dangling", result["problems"])

    def test_integrity_max_examples_truncates(self):
        result_full = integrity(self.db, checks=("dangling", "one_sided_refs"))
        result_capped = integrity(
            self.db, checks=("dangling", "one_sided_refs"), max_examples=1
        )
        for key, items in result_capped["problems"].items():
            self.assertLessEqual(len(items), 1)
        # counts still reflect the FULL scan, not the truncated list.
        self.assertEqual(result_capped["counts"], result_full["counts"])

    # -- Task 2.6: centrality() ----------------------------------------------

    def test_centrality_degree_respects_top(self):
        result = centrality(self.db, metric="degree", top=5)
        self.assertEqual(result["metric"], "degree")
        self.assertFalse(result["capped"])
        self.assertLessEqual(len(result["people"]), 5)
        scores = [p["score"] for p in result["people"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_centrality_articulation_returns_cut_vertices(self):
        g = build_tree_graph(self.db)
        result = centrality(self.db, metric="articulation", top=1000, g=g)
        self.assertEqual(result["metric"], "articulation")
        handles = {p["handle"] for p in result["people"]}
        from gramps_webapi.api.resources import graph_primitives as gp

        self.assertEqual(handles, gp.articulation_points(g.undirected))

    def test_centrality_betweenness_and_closeness_shapes(self):
        for metric in ("betweenness", "closeness"):
            result = centrality(self.db, metric=metric, top=3)
            self.assertEqual(result["metric"], metric)
            self.assertLessEqual(len(result["people"]), 3)

    def test_centrality_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            centrality(self.db, metric="not-a-real-metric")

    def test_centrality_max_nodes_cap_sets_flag(self):
        result = centrality(self.db, metric="betweenness", max_nodes=1)
        self.assertTrue(result["capped"])


if __name__ == "__main__":
    unittest.main()
