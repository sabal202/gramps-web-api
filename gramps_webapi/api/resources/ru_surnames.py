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


def get_patronymic(name):
    """Return the patronymic of a Gramps ``Name``, or "" if there is none.

    In the Russian naming tradition the отчество is stored as a surname entry
    with origintype PATRONYMIC (or MATRONYMIC). Returns the surname string of
    the first such entry so it can be shown as its own column instead of being
    folded into the family surname.
    """
    from gramps.gen.lib import NameOriginType

    want = {NameOriginType.PATRONYMIC, NameOriginType.MATRONYMIC}
    for surn in name.get_surname_list():
        if surn.get_origintype().value in want:
            patronymic = surn.get_surname().strip()
            if patronymic:
                return patronymic
    return ""


def count_unique_surnames(surname_list):
    """Count distinct surnames after collapsing masculine/feminine gender forms.

    ``surname_list`` is the raw list of surname strings from
    ``db.get_surname_list()`` — the first surname of each person. This is fast
    but may include stray patronymics (people whose only/first surname is an
    отчество); prefer :func:`count_unique_family_surnames` for an accurate count.
    Returns the number of distinct gender-normalized surnames.
    """
    return len(
        {normalize_surname_gender(s) for s in surname_list if s and s.strip()}
    )


def count_unique_family_surnames(db, max_scan=20000):
    """Count distinct family surnames in the database, accurately.

    Iterates people and uses :func:`get_family_surname` (patronymic excluded)
    collapsed to the masculine base, so отчества stored as a person's only/first
    surname are not miscounted and gender forms count once. Falls back to the
    cheaper (and slightly looser) ``get_surname_list()`` based count on trees
    larger than ``max_scan`` people to avoid a full scan on huge databases.
    """
    try:
        n_people = db.get_number_of_people()
    except Exception:  # pylint: disable=broad-except
        n_people = 0
    if n_people and n_people <= max_scan:
        seen = set()
        for person in db.iter_people():
            surname = normalize_surname_gender(get_family_surname(person.primary_name))
            if surname:
                seen.add(surname)
        return len(seen)
    return count_unique_surnames(db.get_surname_list() or [])
