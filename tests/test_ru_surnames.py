"""Tests for downstream Russian surname helpers."""

import pytest

from gramps_webapi.api.resources.ru_surnames import (
    count_unique_surnames,
    normalize_surname_gender,
)


@pytest.mark.parametrize(
    "feminine,masculine",
    [
        ("Соболевская", "Соболевский"),
        ("Сабалевская", "Сабалевский"),
        ("Барановская", "Барановский"),
        ("Высоцкая", "Высоцкий"),
        ("Петрова", "Петров"),
        ("Кочева", "Кочев"),
        ("Пушкарёва", "Пушкарёв"),
        ("Лукина", "Лукин"),
        ("Птицына", "Птицын"),
        ("Хисматуллина", "Хисматуллин"),
    ],
)
def test_normalize_feminine_to_masculine(feminine, masculine):
    assert normalize_surname_gender(feminine) == masculine


@pytest.mark.parametrize(
    "surname",
    [
        "Соболевский",  # already masculine
        "Петров",
        "Москаленко",  # indeclinable
        "Сороко",
        "Дурново",
        "Токаев",
    ],
)
def test_normalize_leaves_masculine_and_indeclinable_unchanged(surname):
    assert normalize_surname_gender(surname) == surname


def test_normalize_handles_empty():
    assert normalize_surname_gender("") == ""
    assert normalize_surname_gender(None) is None
    assert normalize_surname_gender("  Петрова ") == "Петров"


def test_count_unique_surnames_collapses_gender():
    raw = ["Соболевский", "Соболевская", "Петров", "Петрова", "Москаленко"]
    # Соболевск*, Петров*, Москаленко -> 3
    assert count_unique_surnames(raw) == 3


def test_count_unique_surnames_ignores_blanks():
    assert count_unique_surnames(["Петров", "", "   ", None, "Петрова"]) == 1


def test_get_family_surname_excludes_patronymic():
    # Requires a gramps environment (constructs real Name/Surname objects).
    gilib = pytest.importorskip("gramps.gen.lib")
    Name = gilib.Name
    Surname = gilib.Surname
    NameOriginType = gilib.NameOriginType

    from gramps_webapi.api.resources.ru_surnames import get_family_surname

    name = Name()
    family = Surname()
    family.set_surname("Соболевский")
    patronymic = Surname()
    patronymic.set_surname("Антонович")
    patronymic.set_origintype(NameOriginType(NameOriginType.PATRONYMIC))
    name.set_surname_list([family, patronymic])

    assert get_family_surname(name) == "Соболевский"


def test_get_family_surname_empty_when_only_patronymic():
    gilib = pytest.importorskip("gramps.gen.lib")
    Name = gilib.Name
    Surname = gilib.Surname
    NameOriginType = gilib.NameOriginType

    from gramps_webapi.api.resources.ru_surnames import get_family_surname

    name = Name()
    patronymic = Surname()
    patronymic.set_surname("Антонович")
    patronymic.set_origintype(NameOriginType(NameOriginType.PATRONYMIC))
    name.set_surname_list([patronymic])

    assert get_family_surname(name) == ""
