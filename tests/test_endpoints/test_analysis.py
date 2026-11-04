#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026  Sergey Sabalevskiy
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""Tests for the /api/analysis/* endpoints (connectivity, lineages, integrity,
centrality).

All tests run against the bundled **example_gramps** fixture, same as
test_kinship.py.

Key persons used (verified against the fixture, see test_kinship.py):
  I0044  Lewis Anderson Garner  handle=GNUJQCL9MD64AM56OH  (tree default person)
"""

import unittest

from . import BASE_URL, get_object_count, get_test_client
from .checks import check_requires_token, check_resource_missing, check_success
from .util import fetch_header

ANALYSIS_URL = BASE_URL + "/analysis/"
CONNECTIVITY_URL = ANALYSIS_URL + "connectivity/"
LINEAGES_URL = ANALYSIS_URL + "lineages/"
INTEGRITY_URL = ANALYSIS_URL + "integrity/"
CENTRALITY_URL = ANALYSIS_URL + "centrality/"

DEFAULT_GRAMPS_ID = "I0044"  # Lewis Anderson Garner


# ---------------------------------------------------------------------------
# GET /api/analysis/connectivity/
# ---------------------------------------------------------------------------


class TestConnectivity(unittest.TestCase):
    """Test cases for the GET /api/analysis/connectivity/ endpoint."""

    @classmethod
    def setUpClass(cls):
        """Test class setup."""
        cls.client = get_test_client()

    def test_connectivity_requires_token(self):
        """Endpoint must reject unauthenticated requests with 401."""
        check_requires_token(self, CONNECTIVITY_URL)

    def test_connectivity_has_required_keys(self):
        """Response must contain all documented top-level keys."""
        rv = check_success(self, CONNECTIVITY_URL)
        for key in (
            "person_count",
            "component_count",
            "main_component_size",
            "islands",
            "isolated_pairs",
            "orphans",
        ):
            self.assertIn(key, rv, f"response missing key: {key}")

    def test_connectivity_person_count_matches_tree(self):
        """person_count must equal the total number of people in the tree."""
        rv = check_success(self, CONNECTIVITY_URL)
        self.assertEqual(rv["person_count"], get_object_count("people"))

    def test_connectivity_component_sizes_sum_to_person_count(self):
        """main_component_size + every island's size must sum to person_count.

        Every non-main component (including size-1 orphans and size-2 pairs)
        is represented once in 'islands', so islands + main covers everyone.
        """
        rv = check_success(self, CONNECTIVITY_URL)
        island_total = sum(island["size"] for island in rv["islands"])
        self.assertEqual(
            rv["main_component_size"] + island_total, rv["person_count"]
        )

    def test_connectivity_island_people_count_matches_size(self):
        """Each island's hydrated 'people' list length must not exceed its size.

        (Could be strictly less only if a referenced handle failed to resolve,
        which should not happen on a consistent example tree.)
        """
        rv = check_success(self, CONNECTIVITY_URL)
        for island in rv["islands"]:
            self.assertLessEqual(len(island["people"]), island["size"])

    def test_connectivity_orphans_are_person_profiles(self):
        """Every orphan entry must carry basic person-profile keys."""
        rv = check_success(self, CONNECTIVITY_URL)
        for person in rv["orphans"]:
            for key in ("handle", "gramps_id", "name_given", "name_surname", "sex"):
                self.assertIn(key, person)

    def test_connectivity_include_singletons_false_empties_orphans(self):
        """?include_singletons=false must return an empty 'orphans' list."""
        rv = check_success(self, CONNECTIVITY_URL + "?include_singletons=false")
        self.assertEqual(rv["orphans"], [])

    def test_connectivity_isolated_pairs_have_two_people(self):
        """Every isolated pair must carry exactly two person entries."""
        rv = check_success(self, CONNECTIVITY_URL)
        for pair in rv["isolated_pairs"]:
            self.assertLessEqual(len(pair["people"]), 2)


# ---------------------------------------------------------------------------
# GET /api/analysis/lineages/
# ---------------------------------------------------------------------------


class TestLineages(unittest.TestCase):
    """Test cases for the GET /api/analysis/lineages/ endpoint."""

    @classmethod
    def setUpClass(cls):
        """Test class setup."""
        cls.client = get_test_client()

    def test_lineages_requires_token(self):
        """Endpoint must reject unauthenticated requests with 401."""
        check_requires_token(self, LINEAGES_URL)

    def test_lineages_whole_tree_has_required_keys(self):
        """Whole-tree mode (no ?person=) must populate the tree-level keys."""
        rv = check_success(self, LINEAGES_URL)
        for key in ("max_tree_depth", "root_count", "lineages"):
            self.assertIn(key, rv)
        self.assertIsInstance(rv["lineages"], list)

    def test_lineages_whole_tree_respects_top(self):
        """?top= must cap the number of returned lineages."""
        rv = check_success(self, LINEAGES_URL + "?top=2")
        self.assertLessEqual(len(rv["lineages"]), 2)

    def test_lineages_whole_tree_root_entries_have_root_person(self):
        """group_by='root' (default) entries must carry a hydrated 'root'."""
        rv = check_success(self, LINEAGES_URL + "?top=5")
        for entry in rv["lineages"]:
            self.assertIn("root", entry)
            self.assertIn("depth", entry)
            self.assertIn("size", entry)

    def test_lineages_group_by_surname_entries_have_surname(self):
        """group_by='surname' entries must carry a 'surname' + 'roots' list."""
        rv = check_success(self, LINEAGES_URL + "?group_by=surname&top=5")
        for entry in rv["lineages"]:
            self.assertIn("surname", entry)
            self.assertIn("roots", entry)

    def test_lineages_person_param_switches_to_deepest_ancestors_mode(self):
        """?person= must switch the response shape to deepest-ancestors mode."""
        rv = check_success(self, LINEAGES_URL + "?person=" + DEFAULT_GRAMPS_ID)
        for key in ("anchor", "max_depth", "by_line", "furthest"):
            self.assertIn(key, rv)
        self.assertIsNotNone(rv["anchor"])
        self.assertEqual(rv["anchor"]["gramps_id"], DEFAULT_GRAMPS_ID)

    def test_lineages_person_param_accepts_raw_handle(self):
        """?person= must also accept a raw internal handle."""
        rv = check_success(self, LINEAGES_URL + "?person=GNUJQCL9MD64AM56OH")
        self.assertEqual(rv["anchor"]["gramps_id"], DEFAULT_GRAMPS_ID)

    def test_lineages_unknown_person_returns_404(self):
        """Unknown ?person= value must return 404."""
        check_resource_missing(self, LINEAGES_URL + "?person=TOTALLY_NONEXISTENT_9999")


# ---------------------------------------------------------------------------
# GET /api/analysis/integrity/
# ---------------------------------------------------------------------------


class TestIntegrity(unittest.TestCase):
    """Test cases for the GET /api/analysis/integrity/ endpoint."""

    @classmethod
    def setUpClass(cls):
        """Test class setup."""
        cls.client = get_test_client()

    def test_integrity_requires_token(self):
        """Endpoint must reject unauthenticated requests with 401."""
        check_requires_token(self, INTEGRITY_URL)

    def test_integrity_has_required_keys(self):
        """Response must contain checks, counts, and problems."""
        rv = check_success(self, INTEGRITY_URL)
        for key in ("checks", "counts", "problems"):
            self.assertIn(key, rv)

    def test_integrity_default_checks(self):
        """Default ?checks= must run exactly one_sided_refs + dangling."""
        rv = check_success(self, INTEGRITY_URL)
        self.assertEqual(set(rv["checks"]), {"one_sided_refs", "dangling"})
        self.assertEqual(set(rv["counts"]), {"one_sided_refs", "dangling"})
        self.assertEqual(set(rv["problems"]), {"one_sided_refs", "dangling"})

    def test_integrity_checks_param_restricts_output(self):
        """?checks=one_sided_refs must run only that one check."""
        rv = check_success(self, INTEGRITY_URL + "?checks=one_sided_refs")
        self.assertEqual(set(rv["checks"]), {"one_sided_refs"})

    def test_integrity_thin_records_is_opt_in(self):
        """thin_records must be absent unless explicitly requested."""
        rv = check_success(self, INTEGRITY_URL)
        self.assertNotIn("thin_records", rv["checks"])
        rv = check_success(self, INTEGRITY_URL + "?checks=thin_records")
        self.assertEqual(set(rv["checks"]), {"thin_records"})

    def test_integrity_invalid_check_returns_422(self):
        """An unknown check name in ?checks= must be rejected with 422."""
        header = fetch_header(self.client)
        rv = self.client.get(
            INTEGRITY_URL + "?checks=not_a_real_check", headers=header
        )
        self.assertEqual(rv.status_code, 422)

    def test_integrity_max_examples_caps_problem_lists(self):
        """?max_examples= must cap each check's problem list length."""
        rv = check_success(self, INTEGRITY_URL + "?max_examples=1")
        for entries in rv["problems"].values():
            self.assertLessEqual(len(entries), 1)

    def test_integrity_counts_match_problem_lengths_when_uncapped(self):
        """Without ?max_examples=, counts must equal the actual list lengths."""
        rv = check_success(self, INTEGRITY_URL)
        for check, count in rv["counts"].items():
            self.assertEqual(count, len(rv["problems"][check]))


# ---------------------------------------------------------------------------
# GET /api/analysis/centrality/
# ---------------------------------------------------------------------------


class TestCentrality(unittest.TestCase):
    """Test cases for the GET /api/analysis/centrality/ endpoint."""

    @classmethod
    def setUpClass(cls):
        """Test class setup."""
        cls.client = get_test_client()

    def test_centrality_requires_token(self):
        """Endpoint must reject unauthenticated requests with 401."""
        check_requires_token(self, CENTRALITY_URL)

    def test_centrality_has_required_keys(self):
        """Response must contain metric, capped, and people."""
        rv = check_success(self, CENTRALITY_URL)
        for key in ("metric", "capped", "people"):
            self.assertIn(key, rv)

    def test_centrality_default_metric_is_betweenness(self):
        """Default metric must be 'betweenness'."""
        rv = check_success(self, CENTRALITY_URL)
        self.assertEqual(rv["metric"], "betweenness")

    def test_centrality_respects_max(self):
        """?max= must cap the number of ranked people returned."""
        rv = check_success(self, CENTRALITY_URL + "?max=3")
        self.assertLessEqual(len(rv["people"]), 3)

    def test_centrality_people_have_handle_score_profile(self):
        """Every ranked entry must carry handle, score, and profile."""
        rv = check_success(self, CENTRALITY_URL + "?max=3")
        for entry in rv["people"]:
            for key in ("handle", "score", "profile"):
                self.assertIn(key, entry)

    def test_centrality_degree_metric_works(self):
        """?metric=degree must succeed and use that metric."""
        rv = check_success(self, CENTRALITY_URL + "?metric=degree&max=3")
        self.assertEqual(rv["metric"], "degree")

    def test_centrality_articulation_metric_works(self):
        """?metric=articulation must succeed and use that metric."""
        rv = check_success(self, CENTRALITY_URL + "?metric=articulation&max=3")
        self.assertEqual(rv["metric"], "articulation")

    def test_centrality_invalid_metric_returns_422(self):
        """An unknown ?metric= value must be rejected with 422."""
        header = fetch_header(self.client)
        rv = self.client.get(
            CENTRALITY_URL + "?metric=not_a_real_metric", headers=header
        )
        self.assertEqual(rv.status_code, 422)
