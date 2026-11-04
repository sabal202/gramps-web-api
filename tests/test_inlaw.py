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

"""Unit tests for the in-law (affinal) relationship helpers.

These exercise the pure graph-walk helpers with lightweight stand-ins for the
Gramps database and objects; no example database or app context is required.
"""

from types import SimpleNamespace

from gramps.gen.lib import EventType, Person

from gramps_webapi.api.resources.inlaw import (
    _children,
    _is_russian,
    _marriage_dissolved,
    _parents,
    _spouses,
)


class FakeLocale:
    def __init__(self, language):
        self.language = language


class FakeDb:
    """Minimal handle->object resolver for the helpers under test."""

    def __init__(self, persons=None, families=None, events=None):
        self.persons = persons or {}
        self.families = families or {}
        self.events = events or {}

    def get_person_from_handle(self, handle):
        return self.persons.get(handle)

    def get_family_from_handle(self, handle):
        return self.families.get(handle)

    def get_event_from_handle(self, handle):
        return self.events.get(handle)


def person(handle, family_handles=(), parent_family_handles=()):
    return SimpleNamespace(
        handle=handle,
        get_family_handle_list=lambda: list(family_handles),
        get_parent_family_handle_list=lambda: list(parent_family_handles),
    )


def family(father=None, mother=None, children=(), event_handles=()):
    return SimpleNamespace(
        get_father_handle=lambda: father,
        get_mother_handle=lambda: mother,
        get_child_ref_list=lambda: [SimpleNamespace(ref=c) for c in children],
        get_event_ref_list=lambda: [SimpleNamespace(ref=e) for e in event_handles],
    )


def event(etype):
    return SimpleNamespace(get_type=lambda: etype)


# --- _is_russian -----------------------------------------------------------


def test_is_russian_true_for_ru():
    assert _is_russian(FakeLocale(["ru"])) is True


def test_is_russian_true_for_ru_region():
    assert _is_russian(FakeLocale(["ru_RU"])) is True


def test_is_russian_false_for_en():
    assert _is_russian(FakeLocale(["en"])) is False


def test_is_russian_false_for_empty_or_none():
    assert _is_russian(FakeLocale([])) is False
    assert _is_russian(FakeLocale(None)) is False


# --- _spouses --------------------------------------------------------------


def test_spouses_returns_partner_when_person_is_father():
    wife = person("W")
    fam = family(father="H", mother="W")
    db = FakeDb(persons={"W": wife}, families={"F1": fam})
    husband = person("H", family_handles=["F1"])
    assert _spouses(db, husband) == [wife]


def test_spouses_returns_partner_when_person_is_mother():
    husband = person("H")
    fam = family(father="H", mother="W")
    db = FakeDb(persons={"H": husband}, families={"F1": fam})
    wife = person("W", family_handles=["F1"])
    assert _spouses(db, wife) == [husband]


def test_spouses_skips_single_parent_family():
    fam = family(father="H", mother=None)
    db = FakeDb(persons={}, families={"F1": fam})
    husband = person("H", family_handles=["F1"])
    assert _spouses(db, husband) == []


# --- _children -------------------------------------------------------------


def test_children_collects_across_families():
    c1, c2 = person("C1"), person("C2")
    fam = family(father="H", mother="W", children=["C1", "C2"])
    db = FakeDb(persons={"C1": c1, "C2": c2}, families={"F1": fam})
    parent = person("H", family_handles=["F1"])
    assert _children(db, parent) == [c1, c2]


# --- _parents --------------------------------------------------------------


def test_parents_returns_both_parents():
    dad, mom = person("D"), person("M")
    fam = family(father="D", mother="M")
    db = FakeDb(persons={"D": dad, "M": mom}, families={"F1": fam})
    kid = person("K", parent_family_handles=["F1"])
    assert _parents(db, kid) == [dad, mom]


# --- _marriage_dissolved ---------------------------------------------------


def test_marriage_dissolved_true_on_divorce():
    fam = family(father="H", mother="W", event_handles=["E1"])
    db = FakeDb(families={"F1": fam}, events={"E1": event(EventType.DIVORCE)})
    husband = person("H", family_handles=["F1"])
    wife = person("W")
    assert _marriage_dissolved(db, husband, wife) is True


def test_marriage_dissolved_true_on_annulment():
    fam = family(father="H", mother="W", event_handles=["E1"])
    db = FakeDb(families={"F1": fam}, events={"E1": event(EventType.ANNULMENT)})
    husband = person("H", family_handles=["F1"])
    wife = person("W")
    assert _marriage_dissolved(db, husband, wife) is True


def test_marriage_dissolved_false_on_marriage_only():
    fam = family(father="H", mother="W", event_handles=["E1"])
    db = FakeDb(families={"F1": fam}, events={"E1": event(EventType.MARRIAGE)})
    husband = person("H", family_handles=["F1"])
    wife = person("W")
    assert _marriage_dissolved(db, husband, wife) is False


def test_marriage_dissolved_false_when_not_spouses():
    fam = family(father="H", mother="W", event_handles=["E1"])
    db = FakeDb(families={"F1": fam}, events={"E1": event(EventType.DIVORCE)})
    husband = person("H", family_handles=["F1"])
    stranger = person("X")
    assert _marriage_dissolved(db, husband, stranger) is False


# --- Person constant sanity (gender prefixes use these) --------------------


def test_person_gender_constants_distinct():
    assert Person.MALE != Person.FEMALE
