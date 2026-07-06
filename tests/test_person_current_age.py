"""Tests for the 'current_age' person-profile computation.

Builds a throwaway sqlite DB with synthetic Person/Event objects and calls
get_person_profile_for_object() directly, bypassing the Flask/marshmallow HTTP
layer entirely (that layer is covered separately in
tests/test_endpoints/test_people.py).
"""

from datetime import date

import pytest
from gramps.cli.clidbman import CLIDbManager
from gramps.gen.const import GRAMPS_LOCALE as glocale
from gramps.gen.db import DbTxn
from gramps.gen.db.utils import make_database
from gramps.gen.dbstate import DbState
from gramps.gen.lib import Date, Event, EventRef, EventType, Name, Person

from gramps_webapi.api.resources.util import get_person_profile_for_object


def _add_event(db, trans, event_type, year, month, day):
    event = Event()
    event.set_type(EventType(event_type))
    date_obj = Date()
    date_obj.set_yr_mon_day(year, month, day)
    event.set_date_object(date_obj)
    return db.add_event(event, trans)


def _make_person(db, trans, given, birth=None, death=None):
    """Create a person with optional (year, month, day) birth/death dates."""
    person = Person()
    name = Name()
    name.set_first_name(given)
    person.set_primary_name(name)

    if birth is not None:
        handle = _add_event(db, trans, EventType.BIRTH, *birth)
        eref = EventRef()
        eref.set_reference_handle(handle)
        person.set_birth_ref(eref)

    if death is not None:
        handle = _add_event(db, trans, EventType.DEATH, *death)
        eref = EventRef()
        eref.set_reference_handle(handle)
        person.set_death_ref(eref)

    person_handle = db.add_person(person, trans)
    return db.get_person_from_handle(person_handle)


@pytest.fixture(scope="module")
def db_handle():
    """Temp sqlite DB with alive / deceased / very-old / no-birth test persons."""
    dbman = CLIDbManager(DbState())
    dirpath, db_name = dbman.create_new_db_cli("_test_current_age", dbid="sqlite")
    db = make_database("sqlite")
    db.load(dirpath)

    today = date.today()
    handles = {}
    with DbTxn("setup", db) as trans:
        # Alive: born exactly 30 years ago today, no death event.
        birth_alive = (today.year - 30, today.month, today.day)
        handles["alive"] = _make_person(db, trans, "Alive", birth=birth_alive).handle

        # Deceased: fixed historical dates -> deterministic 80-year age at death.
        handles["deceased"] = _make_person(
            db, trans, "Deceased", birth=(1900, 1, 1), death=(1980, 1, 1)
        ).handle

        # No death event, but implausibly old -> probably_alive() must be False.
        birth_old = (today.year - 250, today.month, today.day)
        handles["old_no_death"] = _make_person(
            db, trans, "OldNoDeath", birth=birth_old
        ).handle

        # No birth event at all.
        handles["no_birth"] = _make_person(db, trans, "NoBirth").handle

    yield db, handles

    db.close()
    dbman.remove_database(db_name)


def _profile(db, handle):
    person = db.get_person_from_handle(handle)
    return get_person_profile_for_object(
        db, person, args=["current_age"], locale=glocale, precision=1
    )


def test_current_age_absent_when_not_requested(db_handle):
    db, handles = db_handle
    person = db.get_person_from_handle(handles["deceased"])
    profile = get_person_profile_for_object(
        db, person, args=["all"], locale=glocale, precision=1
    )
    assert "current_age" not in profile


def test_current_age_for_alive_person(db_handle):
    db, handles = db_handle
    profile = _profile(db, handles["alive"])
    # NOTE: confirm the exact Span-formatted wording when you run this and adjust
    # the literal if it differs from "30 years" (e.g. plural rules) -- this is a
    # real unknown being resolved by running the test, not a planning error.
    assert profile["current_age"] == "30 years"
    assert profile["death"] == {}


def test_current_age_for_deceased_person_is_would_be_now(db_handle):
    db, handles = db_handle
    profile = _profile(db, handles["deceased"])
    expected_years = date.today().year - 1900
    assert profile["current_age"] == f"{expected_years} years"
    # Age at death (existing, unrelated field) stays independently correct.
    # Requires "age" in args too, since death["age"] is gated separately from
    # current_age (see get_person_profile_for_object).
    person = db.get_person_from_handle(handles["deceased"])
    profile_with_age = get_person_profile_for_object(
        db, person, args=["current_age", "age"], locale=glocale, precision=1
    )
    assert profile_with_age["death"]["age"] == "80 years"


def test_current_age_absent_for_implausibly_old_undated_person(db_handle):
    db, handles = db_handle
    profile = _profile(db, handles["old_no_death"])
    assert "current_age" not in profile


def test_current_age_absent_without_birth_event(db_handle):
    db, handles = db_handle
    profile = _profile(db, handles["no_birth"])
    assert "current_age" not in profile
