#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      David Straub
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

"""Tests for the per-object change history endpoint."""

import os
import unittest
import uuid
from typing import Dict
from unittest.mock import patch

from gramps.cli.clidbman import CLIDbManager
from gramps.gen.dbstate import DbState

from gramps_webapi.app import create_app
from gramps_webapi.auth import add_user, user_db
from gramps_webapi.auth.const import (
    ROLE_CONTRIBUTOR,
    ROLE_EDITOR,
    ROLE_GUEST,
    ROLE_OWNER,
)
from gramps_webapi.const import ENV_CONFIG_FILE, TEST_EMPTY_GRAMPS_AUTH_CONFIG


def get_headers(client, user: str, password: str) -> Dict[str, str]:
    """Get the auth headers for a specific user."""
    rv = client.post("/api/token/", json={"username": user, "password": password})
    access_token = rv.json["access_token"]
    return {"Authorization": "Bearer {}".format(access_token)}


def make_handle() -> str:
    """Make a new valid handle."""
    return str(uuid.uuid4())


class TestObjectHistoryResource(unittest.TestCase):
    def setUp(self):
        self.name = "Test Web API Object History"
        self.dbman = CLIDbManager(DbState())
        dirpath, _ = self.dbman.create_new_db_cli(self.name, dbid="sqlite")
        tree = os.path.basename(dirpath)
        with patch.dict("os.environ", {ENV_CONFIG_FILE: TEST_EMPTY_GRAMPS_AUTH_CONFIG}):
            self.app = create_app(config_from_env=False, config={"TREE": self.name})
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        with self.app.app_context():
            user_db.create_all()
            add_user(name="user", password="123", role=ROLE_GUEST, tree=tree)
            add_user(name="admin", password="123", role=ROLE_OWNER, tree=tree)
            add_user(
                name="contributor", password="123", role=ROLE_CONTRIBUTOR, tree=tree
            )
            add_user(name="editor", password="123", role=ROLE_EDITOR, tree=tree)

    def tearDown(self):
        self.dbman.remove_database(self.name)

    def _build_fixture(self, headers):
        """Create a Person with a referenced Event, and a Family containing them.

        Returns a dict of handles: father, mother, birth_event, marriage_event,
        family. Each object is touched by at least one dedicated transaction,
        so the returned handles can be used to assert scope=object vs.
        scope=page filtering precisely.
        """
        client = self.client
        handle_father = make_handle()
        handle_mother = make_handle()
        handle_birth = make_handle()
        handle_marriage = make_handle()
        handle_family = make_handle()

        # T1: create father (touches: father)
        rv = client.post(
            "/api/people/",
            json={"_class": "Person", "handle": handle_father, "gender": 1},
            headers=headers,
        )
        assert rv.status_code == 201

        # T2: create mother (touches: mother)
        rv = client.post(
            "/api/people/",
            json={"_class": "Person", "handle": handle_mother, "gender": 0},
            headers=headers,
        )
        assert rv.status_code == 201

        # T3: create birth event, unlinked so far (touches: birth event only)
        rv = client.post(
            "/api/events/",
            json={
                "_class": "Event",
                "handle": handle_birth,
                "type": {"_class": "EventType", "string": "Birth"},
            },
            headers=headers,
        )
        assert rv.status_code == 201

        # T4: link birth event to father (touches: father only)
        rv = client.get(f"/api/people/{handle_father}", headers=headers)
        assert rv.status_code == 200
        father = rv.json
        father["event_ref_list"] = [
            {
                "_class": "EventRef",
                "ref": handle_birth,
                "role": {"_class": "EventRoleType", "string": "Primary"},
            }
        ]
        rv = client.put(f"/api/people/{handle_father}", json=father, headers=headers)
        assert rv.status_code == 200

        # T5: create the family (touches: family, and also father+mother,
        # since the backend updates their family_list on commit)
        rv = client.post(
            "/api/families/",
            json={
                "_class": "Family",
                "handle": handle_family,
                "father_handle": handle_father,
                "mother_handle": handle_mother,
            },
            headers=headers,
        )
        assert rv.status_code == 201

        # T6: create marriage event, unlinked so far (touches: marriage event only)
        rv = client.post(
            "/api/events/",
            json={
                "_class": "Event",
                "handle": handle_marriage,
                "type": {"_class": "EventType", "string": "Marriage"},
            },
            headers=headers,
        )
        assert rv.status_code == 201

        # T7: link marriage event to the family (touches: family only)
        rv = client.get(f"/api/families/{handle_family}", headers=headers)
        assert rv.status_code == 200
        family = rv.json
        family["event_ref_list"] = [
            {
                "_class": "EventRef",
                "ref": handle_marriage,
                "role": {"_class": "EventRoleType", "string": "Family"},
            }
        ]
        rv = client.put(f"/api/families/{handle_family}", json=family, headers=headers)
        assert rv.status_code == 200

        return {
            "father": handle_father,
            "mother": handle_mother,
            "birth_event": handle_birth,
            "marriage_event": handle_marriage,
            "family": handle_family,
        }

    def test_scope_object_only_touches_person_handle(self):
        headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(headers)

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/?scope=object",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        transactions = rv.json
        self.assertEqual(rv.headers.pop("X-Total-Count"), str(len(transactions)))
        # T1 (create father), T4 (link birth ref), T5 (family creation
        # touches father via family_list update) = 3 transactions.
        self.assertEqual(len(transactions), 3)
        for transaction in transactions:
            touched = {c["obj_handle"] for c in transaction["changes"]}
            self.assertIn(handles["father"], touched)
            self.assertIn(handles["father"], transaction["matched_handles"])

    def test_scope_page_includes_event_and_family(self):
        headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(headers)

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/?scope=page&pagesize=100",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        transactions = rv.json
        # T1, T3 (birth event create), T4, T5, T6 (marriage event create),
        # T7 (family event link) = 6 transactions.
        self.assertEqual(len(transactions), 6)
        self.assertEqual(rv.headers.pop("X-Total-Count"), "6")

        all_matched = set()
        for transaction in transactions:
            all_matched.update(transaction["matched_handles"])
        self.assertEqual(
            all_matched,
            {
                handles["father"],
                handles["birth_event"],
                handles["family"],
                handles["marriage_event"],
            },
        )

    def test_scope_defaults_to_page(self):
        headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(headers)

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/?pagesize=100",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(len(rv.json), 6)

    def test_family_scope_object_excludes_person(self):
        headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(headers)

        rv = self.client.get(
            f"/api/families/{handles['family']}/history/?scope=object",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        transactions = rv.json
        # T5 (family creation) and T7 (marriage event linked) touch the
        # family handle itself; father/mother creation transactions must
        # not appear.
        self.assertEqual(len(transactions), 2)
        for transaction in transactions:
            touched = {c["obj_handle"] for c in transaction["changes"]}
            self.assertIn(handles["family"], touched)

    def test_pagination(self):
        headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(headers)

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/?scope=page&page=1&pagesize=2",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(len(rv.json), 2)
        self.assertEqual(rv.headers.pop("X-Total-Count"), "6")

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/?scope=page&page=3&pagesize=2",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(len(rv.json), 2)

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/?scope=page&page=4&pagesize=2",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(len(rv.json), 0)

    def test_sort_order(self):
        headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(headers)

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/?scope=object&sort=id",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        ascending_ids = [t["id"] for t in rv.json]
        self.assertEqual(ascending_ids, sorted(ascending_ids))

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/?scope=object&sort=-id",
            headers=headers,
        )
        self.assertEqual(rv.status_code, 200)
        descending_ids = [t["id"] for t in rv.json]
        self.assertEqual(descending_ids, sorted(descending_ids, reverse=True))
        self.assertEqual(ascending_ids, list(reversed(descending_ids)))

    def test_unknown_namespace_404(self):
        headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(headers)
        rv = self.client.get(
            f"/api/bogus/{handles['father']}/history/", headers=headers
        )
        self.assertEqual(rv.status_code, 404)

    def test_missing_handle_404(self):
        headers = get_headers(self.client, "editor", "123")
        self._build_fixture(headers)
        rv = self.client.get(f"/api/people/{make_handle()}/history/", headers=headers)
        self.assertEqual(rv.status_code, 404)

    def test_contributor_forbidden(self):
        editor_headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(editor_headers)

        contributor_headers = get_headers(self.client, "contributor", "123")
        rv = self.client.get(
            f"/api/people/{handles['father']}/history/", headers=contributor_headers
        )
        self.assertEqual(rv.status_code, 403)

    def test_editor_allowed(self):
        editor_headers = get_headers(self.client, "editor", "123")
        handles = self._build_fixture(editor_headers)

        rv = self.client.get(
            f"/api/people/{handles['father']}/history/", headers=editor_headers
        )
        self.assertEqual(rv.status_code, 200)
