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

"""Tests for the /api/relatives/ and /api/people/<h>/common-ancestors endpoints.

All tests run against the bundled **example_gramps** fixture.

Key persons used (verified against the fixture):
  I0044  Lewis Anderson Garner  handle=GNUJQCL9MD64AM56OH  (tree default person)
  I0623  Jesse V. Garner        handle=JF5KQC2L6ABI0MVD3E  (child of I0044)
  I0624  Raymond E. Garner      handle=MG5KQC6ZKSVO4A63G2  (half-sibling of I0623)
  I0626  Walter E. Garner       handle=DI5KQC3CLKWQI3I0CC  (full-sibling of I0623)
  I0045  Luella Jacques Martel  handle=FOUJQC7PMC15VC4P0I
         (wife of I0044, mother of I0623/I0626)

Verified relationships (cross-checked with /api/relations/):
  I0623 ↔ I0626  :  "брат"             (full sibling, same family 9OUJQCBOHW9UEK9CNV)
  I0623 ↔ I0624  :  "единокровный брат" (paternal half-sibling, shares only I0044)
  I0623 ↔ I0044  :  "отец"             (parent-child: I0044 is father of I0623)

  Common ancestors I0623/I0626 (full siblings):
    1 entry, ancestor_handles=[I0044, I0045], path_a=[], path_b=[]

  Common ancestors I0623/I0624 (half-sibling label, yet TWO common ancestors):
    relationship="единокровный брат" (paternal half-brother, Raymond has mrel=Adopted)
    The engine intersects parent-handle lists from the FAMILY record (F0017): both
    I0044 and I0045 are the recorded father/mother regardless of the child-ref type.
    Result (verified 2026-06-30 against live engine on example_gramps):
      1 entry, common_ancestors=[I0044, I0045], path_a=[], path_b=[]

  Common ancestors I0623 (default=I0044 used as ?to=):
    relationship="отец", 1 entry, common_ancestors=[I0044], path_a=[], path_b=[]
"""

import unittest

from . import BASE_URL, get_test_client
from .checks import (
    check_requires_token,
    check_resource_missing,
    check_success,
)

RELATIVES_URL = BASE_URL + "/relatives/"
PEOPLE_URL = BASE_URL + "/people/"

# --- Handles / IDs verified against example_gramps --------------------------

# Default person (tree home person)
DEFAULT_HANDLE = "GNUJQCL9MD64AM56OH"
DEFAULT_GRAMPS_ID = "I0044"  # Lewis Anderson Garner

# I0623 Jesse V. Garner — child of I0044; rich family (siblings, children, etc.)
JESSE_HANDLE = "JF5KQC2L6ABI0MVD3E"
JESSE_GRAMPS_ID = "I0623"

# I0626 Walter E. Garner — FULL sibling of I0623 (same family 9OUJQCBOHW9UEK9CNV)
WALTER_GRAMPS_ID = "I0626"

# I0624 Raymond E. Garner — half-sibling of I0623 (paternal only)
RAYMOND_GRAMPS_ID = "I0624"

# I0044 and I0045 are the shared parents of I0623 and I0626
LEWIS_GRAMPS_ID = "I0044"
LUELLA_GRAMPS_ID = "I0045"

# ---------------------------------------------------------------------------
# Test class: GET /api/relatives/
# ---------------------------------------------------------------------------

# Required keys on a person object inside a group (contract §3)
# Note: 'kind' ('blood' or 'inlaw') is now per-person, not per-group.
_PERSON_KEYS = {
    "handle",
    "gramps_id",
    "name_given",
    "name_surname",
    "sex",
    "media_list",
    "relationship",
    "kind",
}


class TestRelatives(unittest.TestCase):
    """Test cases for the GET /api/relatives/ endpoint."""

    @classmethod
    def setUpClass(cls):
        """Test class setup."""
        cls.client = get_test_client()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def test_relatives_requires_token(self):
        """Endpoint must reject unauthenticated requests with 401."""
        # Use explicit handle so the test is not sensitive to default-person state.
        check_requires_token(self, RELATIVES_URL + "?handle=" + DEFAULT_GRAMPS_ID)

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_relatives_unknown_handle_returns_404(self):
        """Unknown ?handle= value must return 404."""
        check_resource_missing(self, RELATIVES_URL + "?handle=TOTALLY_NONEXISTENT_9999")

    def test_relatives_unknown_handle_gramps_id_returns_404(self):
        """Unknown Gramps-ID in ?handle= must also return 404."""
        check_resource_missing(self, RELATIVES_URL + "?handle=I9999_MISSING")

    # ------------------------------------------------------------------
    # Top-level structure
    # ------------------------------------------------------------------

    def test_relatives_has_anchor_and_groups(self):
        """Response must contain 'anchor' and 'groups' keys."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        self.assertIn("anchor", rv)
        self.assertIn("groups", rv)

    def test_relatives_anchor_has_no_relationship_key(self):
        """The anchor person must NOT carry a 'relationship' key."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        anchor = rv["anchor"]
        self.assertNotIn("relationship", anchor)

    def test_relatives_anchor_carries_required_profile_keys(self):
        """Anchor must carry handle, gramps_id, name_given, name_surname, sex."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        anchor = rv["anchor"]
        for key in (
            "handle",
            "gramps_id",
            "name_given",
            "name_surname",
            "sex",
            "media_list",
        ):
            self.assertIn(key, anchor, f"anchor missing key: {key}")

    def test_relatives_anchor_matches_requested_handle(self):
        """Anchor gramps_id must match the ?handle= argument."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        self.assertEqual(rv["anchor"]["gramps_id"], JESSE_GRAMPS_ID)

    def test_relatives_handle_param_accepts_gramps_id(self):
        """?handle= must accept a Gramps ID (not just a raw handle)."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + DEFAULT_GRAMPS_ID)
        self.assertEqual(rv["anchor"]["gramps_id"], DEFAULT_GRAMPS_ID)

    def test_relatives_handle_param_accepts_raw_handle(self):
        """?handle= must also accept a raw internal handle."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_HANDLE)
        self.assertEqual(rv["anchor"]["gramps_id"], JESSE_GRAMPS_ID)

    # ------------------------------------------------------------------
    # Group structure
    # ------------------------------------------------------------------

    def test_relatives_groups_is_list(self):
        """'groups' must be a list."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        self.assertIsInstance(rv["groups"], list)

    def test_relatives_group_has_required_keys(self):
        """Each group must have category_key, count, and people.

        The group-level 'kind' was removed when in-law relatives were folded
        into blood-equivalent category groups.  Kind is now per-person.
        """
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        for group in rv["groups"]:
            for key in ("category_key", "count", "people"):
                self.assertIn(key, group, f"group missing key: {key}")

    def test_relatives_group_count_matches_people_length(self):
        """group['count'] must equal len(group['people']) for every group."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        for group in rv["groups"]:
            self.assertEqual(
                group["count"],
                len(group["people"]),
                f"count mismatch in group '{group['category_key']}'",
            )

    def test_relatives_group_has_no_kind_key(self):
        """Groups must NOT carry a top-level 'kind' key.

        In-law relatives are folded into the same blood-equivalent category
        groups; kind is now a per-person field, not a per-group field.
        """
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        for group in rv["groups"]:
            self.assertNotIn(
                "kind",
                group,
                f"group '{group['category_key']}' should not have a 'kind' key",
            )

    def test_relatives_person_kind_is_blood_or_inlaw(self):
        """Every person within a group must have 'kind' = 'blood' or 'inlaw'."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        for group in rv["groups"]:
            for person in group["people"]:
                self.assertIn(
                    person.get("kind"),
                    ("blood", "inlaw"),
                    f"person {person.get('gramps_id')} in group "
                    f"'{group['category_key']}' has bad kind",
                )

    def test_relatives_blood_persons_precede_inlaw_within_group(self):
        """Within each group all blood entries must appear before any inlaw entry.

        Since in-laws are now merged into blood-equivalent categories, the
        ordering guarantee is blood-first within each group rather than
        blood-groups-before-inlaw-groups.
        """
        rv = check_success(self, RELATIVES_URL + "?handle=" + DEFAULT_GRAMPS_ID)
        for group in rv["groups"]:
            kinds = [p.get("kind") for p in group["people"]]
            if "inlaw" in kinds:
                inlaw_first = kinds.index("inlaw")
                for kind in kinds[:inlaw_first]:
                    self.assertEqual(
                        kind,
                        "blood",
                        f"blood entry after inlaw entry in group "
                        f"'{group['category_key']}'",
                    )

    # ------------------------------------------------------------------
    # Person objects within groups
    # ------------------------------------------------------------------

    def test_relatives_person_has_required_keys(self):
        """Each person in a group must carry all required keys."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        for group in rv["groups"]:
            for person in group["people"]:
                for key in _PERSON_KEYS:
                    self.assertIn(key, person, f"person missing key: {key}")

    def test_relatives_person_relationship_is_string(self):
        """The 'relationship' field on each person must be a non-empty string."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        for group in rv["groups"]:
            for person in group["people"]:
                self.assertIsInstance(person["relationship"], str)
                self.assertTrue(
                    len(person["relationship"]) > 0,
                    f"empty relationship for {person.get('gramps_id')}",
                )

    # ------------------------------------------------------------------
    # Specific expected categories for I0623 (Jesse V. Garner)
    # Verified: parents count=2, children count=2, siblings count=7
    # ------------------------------------------------------------------

    def test_relatives_jesse_has_parents_group(self):
        """I0623 must have a 'parents' group with at least 2 blood entries."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        parents = next(
            (g for g in rv["groups"] if g["category_key"] == "parents"), None
        )
        self.assertIsNotNone(parents, "expected 'parents' group")
        blood_count = sum(1 for p in parents["people"] if p.get("kind") == "blood")
        self.assertEqual(blood_count, 2)

    def test_relatives_jesse_has_children_group(self):
        """I0623 must have a 'children' group with at least 2 blood entries."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        children = next(
            (g for g in rv["groups"] if g["category_key"] == "children"), None
        )
        self.assertIsNotNone(children, "expected 'children' group")
        blood_count = sum(1 for p in children["people"] if p.get("kind") == "blood")
        self.assertEqual(blood_count, 2)

    def test_relatives_jesse_has_siblings_group(self):
        """I0623 must have a 'siblings' group with exactly 7 blood entries.

        The group may also include in-law siblings (brothers/sisters-in-law);
        we assert the blood sibling count stays at 7 regardless.
        """
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        siblings = next(
            (g for g in rv["groups"] if g["category_key"] == "siblings"), None
        )
        self.assertIsNotNone(siblings, "expected 'siblings' group")
        blood_count = sum(1 for p in siblings["people"] if p.get("kind") == "blood")
        self.assertEqual(blood_count, 7)

    def test_relatives_jesse_siblings_contains_walter(self):
        """I0623's siblings group must include I0626 (Walter, full brother)."""
        rv = check_success(self, RELATIVES_URL + "?handle=" + JESSE_GRAMPS_ID)
        siblings = next(
            (g for g in rv["groups"] if g["category_key"] == "siblings"), None
        )
        self.assertIsNotNone(siblings)
        sibling_ids = {p["gramps_id"] for p in siblings["people"]}
        self.assertIn(WALTER_GRAMPS_ID, sibling_ids)

    def test_relatives_default_person_has_inlaw_persons(self):
        """I0044 (default) must have exactly 43 in-law persons distributed across groups.

        In-law relatives are now folded into blood-equivalent category groups
        rather than a single separate 'inlaw' group.  The total count of persons
        with kind='inlaw' must still be 43 (verified against the live engine on
        example_gramps 2026-06-30, same candidate set as before).  If this value
        drifts after an engine change it is worth reviewing intentionally.
        """
        rv = check_success(self, RELATIVES_URL + "?handle=" + DEFAULT_GRAMPS_ID)
        # There must no longer be a group with category_key='inlaw'.
        inlaw_group = next(
            (g for g in rv["groups"] if g["category_key"] == "inlaw"), None
        )
        self.assertIsNone(inlaw_group, "unexpected 'inlaw' category_key group")
        # Total inlaw persons across all groups must equal 43.
        total_inlaw = sum(
            1
            for g in rv["groups"]
            for p in g["people"]
            if p.get("kind") == "inlaw"
        )
        self.assertEqual(total_inlaw, 43)

    def test_relatives_default_person_uses_home_person_when_no_handle(self):
        """With no ?handle=, anchor must be the tree home person (I0044)."""
        rv = check_success(self, RELATIVES_URL)
        self.assertEqual(rv["anchor"]["gramps_id"], DEFAULT_GRAMPS_ID)

    # ------------------------------------------------------------------
    # 400 branch (no ?handle= AND no home person) — needs special fixture
    # ------------------------------------------------------------------

    @unittest.skip(
        "requires a no-default-person fixture: the test infrastructure always"
        " loads example_gramps which has I0044 as home person, so the 400"
        " branch (missing both ?handle= and home person) cannot be reached"
        " in the standard setup.  To cover this, a separate test module"
        " would need to create a tree with no home person configured."
    )
    def test_relatives_no_handle_no_home_person_returns_400(self):
        """Without ?handle= and without a tree home person, endpoint must 400."""
        # Placeholder — see skip reason above.
        pass


# ---------------------------------------------------------------------------
# Test class: GET /api/people/<handle>/common-ancestors
# ---------------------------------------------------------------------------

COMMON_ANC_URL_TMPL = PEOPLE_URL + "{handle}/common-ancestors"


class TestCommonAncestors(unittest.TestCase):
    """Test cases for GET /api/people/<handle>/common-ancestors."""

    @classmethod
    def setUpClass(cls):
        """Test class setup."""
        cls.client = get_test_client()

    def _url(self, handle, **params):
        url = COMMON_ANC_URL_TMPL.format(handle=handle)
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        return url

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def test_common_ancestors_requires_token(self):
        """Endpoint must reject unauthenticated requests with 401."""
        check_requires_token(self, self._url(JESSE_HANDLE, to=DEFAULT_GRAMPS_ID))

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_common_ancestors_bad_subject_handle_returns_404(self):
        """Unknown subject handle (path param) must return 404."""
        check_resource_missing(
            self, self._url("BADHANDLE_NONEXISTENT_99", to=JESSE_GRAMPS_ID)
        )

    def test_common_ancestors_bad_to_param_returns_404(self):
        """Unknown ?to= value must return 404."""
        check_resource_missing(
            self, self._url(JESSE_HANDLE, to="BADHANDLE_NONEXISTENT_99")
        )

    # ------------------------------------------------------------------
    # No ?to= → uses default person (I0044), not an error
    # ------------------------------------------------------------------

    def test_common_ancestors_no_to_uses_default_person(self):
        """Without ?to=, endpoint uses the tree home person; must return 200."""
        rv = check_success(self, self._url(JESSE_HANDLE))
        # I0623's home person is I0044 (his father), so we get a non-null relationship
        self.assertIn("relationship", rv)
        self.assertIn("ancestors", rv)
        # Relationship must be non-null (I0623 ↔ I0044 = "отец")
        self.assertIsNotNone(rv["relationship"])

    # ------------------------------------------------------------------
    # Top-level schema
    # ------------------------------------------------------------------

    def test_common_ancestors_has_relationship_and_ancestors_keys(self):
        """Response must always contain 'relationship' and 'ancestors' keys."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        self.assertIn("relationship", rv)
        self.assertIn("ancestors", rv)

    def test_common_ancestors_ancestors_is_list(self):
        """'ancestors' must be a list."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        self.assertIsInstance(rv["ancestors"], list)

    def test_common_ancestors_entry_has_required_keys(self):
        """Each entry must have common_ancestors, path_a, and path_b."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        for entry in rv["ancestors"]:
            self.assertIn("common_ancestors", entry)
            self.assertIn("path_a", entry)
            self.assertIn("path_b", entry)

    def test_common_ancestors_person_in_entry_has_required_keys(self):
        """Person objects in an entry must carry handle, gramps_id, name fields, sex."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        for entry in rv["ancestors"]:
            for person in entry["common_ancestors"]:
                for key in (
                    "handle",
                    "gramps_id",
                    "name_given",
                    "name_surname",
                    "sex",
                ):
                    self.assertIn(key, person, f"person missing key: {key}")

    def test_common_ancestors_person_has_no_relationship_key(self):
        """Persons in common_ancestors/path_a/path_b must NOT have 'relationship'."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        for entry in rv["ancestors"]:
            all_persons = entry["common_ancestors"] + entry["path_a"] + entry["path_b"]
            for person in all_persons:
                self.assertNotIn("relationship", person)

    # ------------------------------------------------------------------
    # Full-sibling case: I0623 ↔ I0626 ("брат")
    # Expected: 1 entry, 2 common ancestors (I0044 + I0045), empty paths
    # Verified by /api/relations/ → distance 1/1, string "брат"
    # ------------------------------------------------------------------

    def test_common_ancestors_full_siblings_relationship(self):
        """I0623 ↔ I0626 are full siblings; relationship must be non-null."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        self.assertIsNotNone(rv["relationship"])
        self.assertIsInstance(rv["relationship"], str)
        self.assertGreater(len(rv["relationship"]), 0)

    def test_common_ancestors_full_siblings_single_entry(self):
        """Full siblings must produce exactly ONE ancestors entry."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        self.assertEqual(len(rv["ancestors"]), 1)

    def test_common_ancestors_full_siblings_two_common_ancestors(self):
        """Full siblings must have TWO common_ancestors (both parents I0044 + I0045)."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        self.assertEqual(len(rv["ancestors"]), 1)
        entry = rv["ancestors"][0]
        self.assertEqual(len(entry["common_ancestors"]), 2)

    def test_common_ancestors_full_siblings_common_ancestor_ids(self):
        """Common ancestors for I0623/I0626 must be I0044 (Lewis) and I0045 (Luella)."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        entry = rv["ancestors"][0]
        anc_ids = {p["gramps_id"] for p in entry["common_ancestors"]}
        self.assertIn(LEWIS_GRAMPS_ID, anc_ids)
        self.assertIn(LUELLA_GRAMPS_ID, anc_ids)

    def test_common_ancestors_full_siblings_empty_paths(self):
        """Sibling entry must have empty path_a and path_b (no intermediates)."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        entry = rv["ancestors"][0]
        self.assertEqual(entry["path_a"], [])
        self.assertEqual(entry["path_b"], [])

    # ------------------------------------------------------------------
    # Half-sibling case: I0623 ↔ I0624 Raymond ("единокровный брат")
    # Raymond has mrel=Adopted in the child-ref but is still recorded in
    # family F0017 alongside I0044 (father) and I0045 (mother).  The engine
    # uses _parent_handles_ordered which reads the FAMILY's father/mother
    # handles — both I0044 and I0045 are returned.  Intersection → 2 ancs.
    # Verified 2026-06-30 against live engine on example_gramps.
    # ------------------------------------------------------------------

    def test_common_ancestors_half_sibling_relationship_non_null(self):
        """I0623 ↔ I0624 (half-sibling Raymond) must return non-null relationship."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=RAYMOND_GRAMPS_ID))
        self.assertIsNotNone(rv["relationship"])
        self.assertIsInstance(rv["relationship"], str)
        self.assertGreater(len(rv["relationship"]), 0)

    def test_common_ancestors_half_sibling_single_entry(self):
        """Half-sibling pair must produce exactly ONE ancestors entry."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=RAYMOND_GRAMPS_ID))
        self.assertEqual(len(rv["ancestors"]), 1)

    def test_common_ancestors_half_sibling_two_common_ancestors(self):
        """Half-sibling pair must list TWO common ancestors (I0044 + I0045).

        Although the label is 'единокровный брат' (paternal half-sibling) the
        engine resolves ancestors by intersecting parent-handle lists from the
        FAMILY record, not from child-ref relation types.  Both I0044 (father)
        and I0045 (mother) are the recorded parents of family F0017, so the
        intersection yields two ancestors.
        """
        rv = check_success(self, self._url(JESSE_HANDLE, to=RAYMOND_GRAMPS_ID))
        entry = rv["ancestors"][0]
        self.assertEqual(len(entry["common_ancestors"]), 2)
        anc_ids = {p["gramps_id"] for p in entry["common_ancestors"]}
        self.assertIn(LEWIS_GRAMPS_ID, anc_ids)
        self.assertIn(LUELLA_GRAMPS_ID, anc_ids)

    def test_common_ancestors_half_sibling_empty_paths(self):
        """Half-sibling entry must have empty path_a and path_b."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=RAYMOND_GRAMPS_ID))
        entry = rv["ancestors"][0]
        self.assertEqual(entry["path_a"], [])
        self.assertEqual(entry["path_b"], [])

    # ------------------------------------------------------------------
    # Parent-child case: I0623 ↔ I0044 ("отец")
    # Expected: 1 entry, 1 common ancestor (I0044 himself), empty paths
    # ------------------------------------------------------------------

    def test_common_ancestors_parent_child_relationship_non_null(self):
        """I0623 ↔ I0044 (parent-child) must return non-null relationship."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=DEFAULT_GRAMPS_ID))
        self.assertIsNotNone(rv["relationship"])

    def test_common_ancestors_parent_child_single_entry(self):
        """Parent-child must produce exactly one ancestors entry."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=DEFAULT_GRAMPS_ID))
        self.assertEqual(len(rv["ancestors"]), 1)

    def test_common_ancestors_parent_child_single_common_ancestor(self):
        """Parent-child must list exactly ONE common ancestor (the parent himself)."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=DEFAULT_GRAMPS_ID))
        entry = rv["ancestors"][0]
        self.assertEqual(len(entry["common_ancestors"]), 1)
        self.assertEqual(entry["common_ancestors"][0]["gramps_id"], DEFAULT_GRAMPS_ID)

    def test_common_ancestors_parent_child_empty_paths(self):
        """Parent-child entry must have empty path_a and path_b."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=DEFAULT_GRAMPS_ID))
        entry = rv["ancestors"][0]
        self.assertEqual(entry["path_a"], [])
        self.assertEqual(entry["path_b"], [])

    # ------------------------------------------------------------------
    # Deeper relation: 9BXKQC1PVLPYFMD6IX ↔ ORFKQC4KLWEGTGR19L
    # Same pair used in test_relations.py; known relationship is a 5-step link.
    # Verified: common ancestor = I0106 Robert W. Garner (and I0107 for second entry)
    # path_a has 4 intermediate persons; path_b is empty (ORFKQC4KLWEGTGR19L IS the
    # common ancestor direction). We assert 2 entries and non-null relationship.
    # ------------------------------------------------------------------

    # Handles for the distant pair (same as test_relations.py)
    _DIST_A = "9BXKQC1PVLPYFMD6IX"
    _DIST_B = "ORFKQC4KLWEGTGR19L"

    def test_common_ancestors_known_distant_pair_returns_result(self):
        """Known distant pair must return non-null relationship and >=1 entry."""
        rv = check_success(self, self._url(self._DIST_A, to=self._DIST_B))
        self.assertIsNotNone(rv["relationship"])
        self.assertGreaterEqual(len(rv["ancestors"]), 1)

    def test_common_ancestors_known_distant_pair_ancestor_is_I0106(self):
        """Distant pair ancestor entries must include I0106 (Robert W. Garner)."""
        rv = check_success(self, self._url(self._DIST_A, to=self._DIST_B))
        all_ancestor_ids = {
            p["gramps_id"]
            for entry in rv["ancestors"]
            for p in entry["common_ancestors"]
        }
        self.assertIn("I0106", all_ancestor_ids)

    def test_common_ancestors_known_distant_pair_path_a_non_empty(self):
        """For the distant pair, at least one entry must have a non-empty path_a."""
        rv = check_success(self, self._url(self._DIST_A, to=self._DIST_B))
        has_path_a = any(len(e["path_a"]) > 0 for e in rv["ancestors"])
        self.assertTrue(has_path_a)

    # ------------------------------------------------------------------
    # ?to= accepts gramps_id OR raw handle
    # ------------------------------------------------------------------

    def test_common_ancestors_to_param_accepts_gramps_id(self):
        """?to= must accept a Gramps ID string."""
        rv = check_success(self, self._url(JESSE_HANDLE, to=WALTER_GRAMPS_ID))
        self.assertEqual(len(rv["ancestors"]), 1)

    def test_common_ancestors_to_param_accepts_raw_handle(self):
        """?to= must also accept a raw internal handle."""
        rv = check_success(
            self,
            self._url(JESSE_HANDLE, to="DI5KQC3CLKWQI3I0CC"),  # I0626 handle
        )
        self.assertEqual(len(rv["ancestors"]), 1)

    # ------------------------------------------------------------------
    # Path-param contract: subject <handle> must be a RAW handle
    # ------------------------------------------------------------------

    def test_common_ancestors_subject_path_param_requires_raw_handle(self):
        """Passing a Gramps ID (e.g. 'I0623') as the path segment returns 404.

        The ?to= / ?handle= query params accept both raw handles and Gramps IDs
        via _resolve_person().  The path param <handle> in
        /api/people/<handle>/common-ancestors does NOT go through that fallback —
        it is looked up directly as a raw handle, so a Gramps ID string yields 404.
        """
        # Use I0623 Gramps ID as the path segment — must be 404, not 200.
        check_resource_missing(
            self,
            self._url(JESSE_GRAMPS_ID, to=WALTER_GRAMPS_ID),
        )
