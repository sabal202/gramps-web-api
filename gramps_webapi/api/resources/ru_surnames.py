#
# Gramps Web API - downstream (Russian-tree specific) surname helpers.
#
# In the Russian naming tradition the patronymic is stored as an additional
# surname entry (origintype PATRONYMIC/MATRONYMIC). It is not a family name, so
# it must not be glued onto the "surname" shown in people lists, used as a sort
# key, or counted as a distinct surname. In addition, the same family surname
# has distinct masculine and feminine grammatical forms (Соболевский /
# Соболевская) which should count as one surname.
#
# This module is downstream-only (region specific) and is intentionally kept
# free of gramps imports at module load time so the pure gender-normalization
# helper can be unit-tested without a gramps environment.
#

"""Downstream helpers for Russian surname handling."""

# Feminine surname endings mapped to their masculine base. Order matters:
# more specific endings (ская/цкая, ёва) must be tried before the generic
# ones (ая, ева). Indeclinable or foreign surnames match nothing and are
# returned unchanged.
_FEMININE_ENDINGS = (
    ("ская", "ский"),
    ("цкая", "цкий"),
    ("ёва", "ёв"),
    ("ова", "ов"),
    ("ева", "ев"),
    ("ына", "ын"),
    ("ина", "ин"),
    ("ая", "ий"),
)


def normalize_surname_gender(surname):
    """Return the masculine base of a Russian feminine surname.

    Maps e.g. "Соболевская" -> "Соболевский", "Петрова" -> "Петров" so that
    both grammatical genders collapse to a single surname when counting unique
    surnames. Indeclinable (Москаленко) or foreign surnames match no ending and
    are returned unchanged. Empty/None input is returned as-is.
    """
    if not surname:
        return surname
    stripped = surname.strip()
    for feminine, masculine in _FEMININE_ENDINGS:
        if stripped.endswith(feminine):
            return stripped[: -len(feminine)] + masculine
    return stripped


def get_family_surname(name):
    """Return the formatted surname of a Gramps ``Name``, excluding patronymics.

    Mirrors ``SurnameBase.get_surname()`` (prefix + surname + connector, space
    joined) but skips surname entries whose origintype is PATRONYMIC or
    MATRONYMIC. Used for the surname column, sort keys and profile name_surname
    so an отчество is never presented as a family name. If a person has only a
    patronymic entry (no real surname) an empty string is returned.
    """
    from gramps.gen.lib import NameOriginType

    skip = {NameOriginType.PATRONYMIC, NameOriginType.MATRONYMIC}
    parts = []
    for surn in name.get_surname_list():
        if surn.get_origintype().value in skip:
            continue
        fsurn = surn.get_surname()
        if surn.get_prefix():
            fsurn = f"{surn.get_prefix()} {fsurn}".strip()
        if surn.get_connector():
            fsurn = f"{fsurn} {surn.get_connector()}".strip()
        fsurn = fsurn.strip()
        if fsurn:
            parts.append(fsurn)
    return " ".join(parts).strip()


def count_unique_surnames(surname_list):
    """Count distinct surnames after collapsing masculine/feminine gender forms.

    ``surname_list`` is the raw list of surname strings from
    ``db.get_surname_list()`` (already patronymic-free — it is the first surname
    of each person). Returns the number of distinct gender-normalized surnames.
    """
    return len(
        {normalize_surname_gender(s) for s in surname_list if s and s.strip()}
    )
