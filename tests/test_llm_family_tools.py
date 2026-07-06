#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      Sergey Sabalevskiy
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

"""Tests for the home-person / kinship / anniversaries / statistics LLM tools."""

import os
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from gramps_webapi.api.llm.deps import AgentDeps
from gramps_webapi.api.llm.tools import (
    get_anniversaries,
    get_home_person,
    get_relationship,
    get_relatives,
    get_tree_statistics,
)
from gramps_webapi.api.resources.anniversaries import (
    _next_anniversary_date,
    upcoming_anniversaries,
)
from gramps_webapi.api.util import get_db_outside_request
from gramps_webapi.app import create_app
from gramps_webapi.auth import add_user, user_db
from gramps_webapi.auth.const import ROLE_OWNER
from gramps_webapi.const import ENV_CONFIG_FILE, TEST_EXAMPLE_GRAMPS_AUTH_CONFIG
from gramps_webapi.dbmanager import WebDbManager
from tests import ExampleDbSQLite


TEST_APP = None
TEST_TREE = None
HOME_GID = None  # gramps_id of the example tree's default person


def setUpModule():
    """Create the app + example tree once for the module."""
    global TEST_APP, TEST_TREE, HOME_GID

    test_db = ExampleDbSQLite(name="example_gramps")

    with patch.dict("os.environ", {ENV_CONFIG_FILE: TEST_EXAMPLE_GRAMPS_AUTH_CONFIG}):
        TEST_APP = create_app(
            config={
                "TESTING": True,
                "RATELIMIT_ENABLED": False,
                "MEDIA_BASE_DIR": f"{os.environ['GRAMPS_RESOURCES']}/doc/gramps/example/gramps",
                "LLM_MODEL": "mock-model",
            },
            config_from_env=False,
        )

    with TEST_APP.app_context():
        user_db.create_all()
        db_manager = WebDbManager(name=test_db.name, create_if_missing=True)
        TEST_TREE = db_manager.dirname
        add_user(
            name="test_user",
            password="test_password",
            role=ROLE_OWNER,
            tree=TEST_TREE,
        )
        db = get_db_outside_request(
            tree=TEST_TREE, view_private=True, readonly=True, user_id="test_user"
        )
        try:
            default_person = db.get_default_person()
            HOME_GID = default_person.gramps_id if default_person else None
        finally:
            db.close()


def _ctx(home=None, include_private=True):
    """Build a mock RunContext with AgentDeps."""
    ctx = MagicMock()
    ctx.deps = AgentDeps(
        tree=TEST_TREE,
        include_private=include_private,
        max_context_length=50000,
        user_id="test_user",
        home_person_gramps_id=home,
    )
    return ctx


class TestNextAnniversaryDate(unittest.TestCase):
    """Pure unit tests for the recurrence date helper (no DB)."""

    def test_same_year_future(self):
        self.assertEqual(
            _next_anniversary_date(date(2026, 1, 1), 6, 15), date(2026, 6, 15)
        )

    def test_rolls_to_next_year(self):
        self.assertEqual(
            _next_anniversary_date(date(2026, 7, 1), 3, 10), date(2027, 3, 10)
        )

    def test_today_is_the_day(self):
        self.assertEqual(
            _next_anniversary_date(date(2026, 5, 5), 5, 5), date(2026, 5, 5)
        )

    def test_leap_day_in_non_leap_year_observed_mar_1(self):
        # 2027 is not a leap year -> Feb 29 observed on Mar 1.
        self.assertEqual(
            _next_anniversary_date(date(2027, 1, 1), 2, 29), date(2027, 3, 1)
        )

    def test_leap_day_in_leap_year(self):
        self.assertEqual(
            _next_anniversary_date(date(2028, 1, 1), 2, 29), date(2028, 2, 29)
        )


class TestGetHomePerson(unittest.TestCase):
    """Tests for the get_home_person tool."""

    def _run(self, ctx):
        with TEST_APP.app_context():
            return get_home_person(ctx)

    def test_no_home_person(self):
        result = self._run(_ctx(home=None))
        self.assertIn("No home person", result)

    def test_invalid_home_person(self):
        result = self._run(_ctx(home="I-DOES-NOT-EXIST"))
        self.assertIn("not found", result)

    @unittest.skipIf(HOME_GID is None, "example tree has no default person")
    def test_valid_home_person(self):
        result = self._run(_ctx(home=HOME_GID))
        self.assertIn(HOME_GID, result)
        self.assertNotIn("No home person", result)


class TestGetRelatives(unittest.TestCase):
    """Tests for the get_relatives tool."""

    def _run(self, ctx, **kwargs):
        with TEST_APP.app_context():
            return get_relatives(ctx, **kwargs)

    def test_no_anchor_no_home(self):
        result = self._run(_ctx(home=None))
        self.assertIn("no home person", result.lower())

    def test_invalid_gramps_id(self):
        result = self._run(_ctx(), gramps_id="I-NOPE")
        self.assertIn("No person found", result)

    @unittest.skipIf(HOME_GID is None, "example tree has no default person")
    def test_relatives_of_home_person(self):
        # Omitting gramps_id falls back to the home person.
        result = self._run(_ctx(home=HOME_GID))
        self.assertNotIn("Error", result)
        self.assertIn("Relatives of", result)

    @unittest.skipIf(HOME_GID is None, "example tree has no default person")
    def test_relatives_explicit_id(self):
        result = self._run(_ctx(), gramps_id=HOME_GID)
        self.assertNotIn("Error", result)


class TestGetRelationship(unittest.TestCase):
    """Tests for the get_relationship tool."""

    def _run(self, ctx, a, b=""):
        with TEST_APP.app_context():
            return get_relationship(ctx, a, b)

    def test_missing_first(self):
        result = self._run(_ctx(), a="")
        self.assertIn("gramps_id_a", result)

    def test_no_second_no_home(self):
        result = self._run(_ctx(home=None), a=(HOME_GID or "I0044"))
        self.assertIn("home person", result.lower())

    @unittest.skipIf(HOME_GID is None, "example tree has no default person")
    def test_same_person(self):
        result = self._run(_ctx(), a=HOME_GID, b=HOME_GID)
        self.assertIn("same person", result)

    @unittest.skipIf(HOME_GID is None, "example tree has no default person")
    def test_relationship_with_a_relative(self):
        # Find a blood relative of the home person, then ask the relationship.
        with TEST_APP.app_context():
            rel_text = get_relatives(_ctx(), gramps_id=HOME_GID)
        # Extract a /person/<id> link that is not the home person itself.
        import re

        ids = re.findall(r"/person/([A-Za-z0-9]+)", rel_text)
        other = next((i for i in ids if i != HOME_GID), None)
        if other is None:
            self.skipTest("home person has no linked relatives in example tree")
        result = self._run(_ctx(), a=HOME_GID, b=other)
        self.assertNotIn("Error computing", result)
        # Either a relationship label or an explicit "no relationship" note.
        self.assertTrue(len(result) > 0)


class TestGetAnniversaries(unittest.TestCase):
    """Tests for the get_anniversaries tool + pure helper."""

    def test_tool_all_scope(self):
        with TEST_APP.app_context():
            result = get_anniversaries(_ctx(), within_days=366, scope="all")
        self.assertNotIn("Error", result)

    def test_tool_home_scope_without_home(self):
        with TEST_APP.app_context():
            result = get_anniversaries(_ctx(home=None), within_days=366, scope="home")
        self.assertNotIn("Error", result)
        self.assertIn("whole tree", result)

    def test_pure_helper_returns_events_within_window(self):
        with TEST_APP.app_context():
            db = get_db_outside_request(
                tree=TEST_TREE, view_private=True, readonly=True, user_id="test_user"
            )
            try:
                # Full-year window from a fixed date should surface births.
                results = upcoming_anniversaries(
                    db,
                    within_days=366,
                    today=date(2026, 1, 1),
                    event_types=["Birth"],
                )
            finally:
                db.close()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        # Sorted ascending by upcoming date.
        dates = [r["date"] for r in results]
        self.assertEqual(dates, sorted(dates))


class TestGetTreeStatistics(unittest.TestCase):
    """Tests for the get_tree_statistics tool."""

    def test_statistics(self):
        with TEST_APP.app_context():
            result = get_tree_statistics(_ctx())
        self.assertNotIn("Error", result)
        self.assertIn("People:", result)
        self.assertIn("Families:", result)
        self.assertIn("surnames", result.lower())


if __name__ == "__main__":
    unittest.main()
