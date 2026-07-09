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

"""Pure unit tests for ``get_page_reference_handles``.

Uses fabricated duck-typed stand-ins for Gramps lib objects (plain classes
exposing only the accessor methods the function reads: ``get_media_list``,
``get_note_list``, ``get_citation_list``, ``get_event_ref_list``,
``get_family_handle_list``, ``get_parent_family_handle_list``) instead of
real ``gramps.gen.lib`` instances.

This is deliberate, not a simplification: importing
``gramps_webapi.api.resources.util`` (where ``get_page_reference_handles``
lives, per the design doc) pulls in ``gramps.gen.lib`` at module load time,
which in turn requires the ``gi``/PyGObject bindings. Those aren't available
outside the project's gi-enabled container (same constraint documented for
the in-law tests), so this module uses ``pytest.importorskip`` to skip
gracefully rather than fail collection when ``gi`` is missing - e.g. on a
plain Windows checkout. Where ``gi`` *is* available, the test runs for real
and exercises the actual per-type traversal logic; no DB/backend is needed
either way since the function is only ever given already-constructed
objects and a small getter-based fake "db".
"""

import pytest

util = pytest.importorskip(
    "gramps_webapi.api.resources.util",
    reason="requires a gi/PyGObject-enabled Gramps environment",
)

get_page_reference_handles = util.get_page_reference_handles


class FakeRef:
    """Stand-in for EventRef / MediaRef (anything with a `.ref` handle)."""

    def __init__(self, ref):
        self.ref = ref


class FakeObj:
    """Stand-in for a Gramps primary object.

    Only exposes the accessor methods actually used by
    ``get_page_reference_handles`` (chosen per fabricated "type"), mirroring
    how real Gramps lib objects mix in CitationBase/NoteBase/MediaBase/
    EventBase depending on class.
    """

    def __init__(
        self,
        media=None,
        notes=None,
        citations=None,
        event_refs=None,
        family_handles=None,
        parent_family_handles=None,
    ):
        if media is not None:
            self.get_media_list = lambda: [FakeRef(h) for h in media]
        if notes is not None:
            self.get_note_list = lambda: list(notes)
        if citations is not None:
            self.get_citation_list = lambda: list(citations)
        if event_refs is not None:
            self.get_event_ref_list = lambda: [FakeRef(h) for h in event_refs]
        if family_handles is not None:
            self.get_family_handle_list = lambda: list(family_handles)
        if parent_family_handles is not None:
            self.get_parent_family_handle_list = lambda: list(parent_family_handles)


class FakeDb:
    """Fake db exposing get_<class>_from_handle(handle) getters."""

    def __init__(self, objects_by_class):
        # objects_by_class: {"person": {handle: obj}, "family": {...}, ...}
        self._objects_by_class = objects_by_class

    def __getattr__(self, name):
        if name.startswith("get_") and name.endswith("_from_handle"):
            class_name = name[len("get_") : -len("_from_handle")]

            def getter(handle):
                return self._objects_by_class.get(class_name, {}).get(handle)

            return getter
        raise AttributeError(name)


def test_note_has_no_extra_handles():
    """Note (and equally Tag) have no relevant mixins: self only."""
    db = FakeDb({"note": {"N1": FakeObj()}})
    handles = get_page_reference_handles(db, "Note", "N1")
    assert handles == {"N1"}


def test_event_gathers_media_notes_citations():
    db = FakeDb(
        {
            "event": {
                "E1": FakeObj(media=["M1"], notes=["O1"], citations=["C1"]),
            }
        }
    )
    handles = get_page_reference_handles(db, "Event", "E1")
    assert handles == {"E1", "M1", "O1", "C1"}


def test_repository_gathers_only_notes():
    # Repository has NoteBase but not MediaBase/CitationBase in the real lib;
    # a fixture that only sets notes should yield exactly {handle, notes}.
    db = FakeDb({"repository": {"R1": FakeObj(notes=["O1"])}})
    handles = get_page_reference_handles(db, "Repository", "R1")
    assert handles == {"R1", "O1"}


def test_source_gathers_media_and_notes_not_citations():
    # Source has MediaBase + NoteBase but not CitationBase in the real lib.
    db = FakeDb({"source": {"S1": FakeObj(media=["M1"], notes=["O1"])}})
    handles = get_page_reference_handles(db, "Source", "S1")
    assert handles == {"S1", "M1", "O1"}


def test_family_gathers_media_notes_citations_and_events():
    db = FakeDb(
        {
            "family": {
                "F1": FakeObj(
                    media=["M1"], notes=["O1"], citations=["C1"], event_refs=["E1"]
                ),
            },
            "event": {"E1": FakeObj(media=["M2"])},
        }
    )
    handles = get_page_reference_handles(db, "Family", "F1")
    # F1's own refs, plus E1 (from event_ref_list), plus E1's own media M2.
    assert handles == {"F1", "M1", "O1", "C1", "E1", "M2"}


def test_person_gathers_own_refs_events_and_families():
    db = FakeDb(
        {
            "person": {
                "P1": FakeObj(
                    media=["M1"],
                    notes=["O1"],
                    citations=["C1"],
                    event_refs=["E1"],
                    family_handles=["F1"],
                    parent_family_handles=["F2"],
                ),
            },
            "event": {"E1": FakeObj(notes=["O2"])},
            "family": {
                "F1": FakeObj(event_refs=["E2"]),
                "F2": FakeObj(media=["M2"]),
            },
        }
    )
    # Add F1's referenced event E2 into the "event" bucket too.
    db._objects_by_class["event"]["E2"] = FakeObj(citations=["C2"])

    handles = get_page_reference_handles(db, "Person", "P1")
    assert handles == {
        "P1",  # self
        "M1",
        "O1",
        "C1",  # own media/notes/citations
        "E1",  # own event
        "O2",  # E1's own note
        "F1",
        "F2",  # own families (family_list + parent_family_list)
        "E2",  # F1's own event
        "C2",  # E2's own citation
        "M2",  # F2's own media
    }


def test_missing_object_returns_just_the_handle():
    db = FakeDb({"person": {}})
    handles = get_page_reference_handles(db, "Person", "MISSING")
    assert handles == {"MISSING"}


def test_falsy_refs_are_skipped():
    db = FakeDb(
        {
            "person": {
                "P1": FakeObj(
                    media=[None, "M1"],
                    notes=["", "O1"],
                    citations=[None],
                    event_refs=[None, "E1"],
                    family_handles=[None, ""],
                    parent_family_handles=[],
                ),
            },
            "event": {"E1": FakeObj()},
        }
    )
    handles = get_page_reference_handles(db, "Person", "P1")
    assert handles == {"P1", "M1", "O1", "E1"}
