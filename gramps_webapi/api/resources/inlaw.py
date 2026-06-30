#
# Gramps Web API - in-law (affinal) relationship fallback
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

"""Affinal (in-law) relationship fallback for Gramps Web API.

The core Gramps relationship calculator resolves only consanguineal (blood)
relationships and direct spouses; affinal ties (mother-in-law, son-in-law,
brother-in-law, ...) come back empty because two in-laws share no common
ancestor.

This module fills that gap with a cheap, bounded, single-marriage search:
it reuses the existing calculator to find a blood relationship between the
queried person and the *spouse* of one of the two people, then renders it
with the ``in_law_a`` / ``in_law_b`` flags the calculator already supports.

The blood-tie rendering is locale-general (every localized calculator
translates the in_law flags). Two embellishments are not expressible through
the calculator and are hand-written in Russian only -- the "former spouse"
qualifier and the сват/сватья (co-parents-in-law) term; both are gated on the
Russian locale and simply omitted elsewhere.
"""

from __future__ import annotations

from gramps.gen.db.base import DbReadBase
from gramps.gen.lib import EventType, Person
from gramps.gen.relationship import RelationshipCalculator, get_relationship_calculator
from gramps.gen.utils.grampslocale import GrampsLocale

# Cap the marriage-bridge blood search; in-laws of interest are always close.
_MAX_DEPTH = 5


def _is_russian(locale: GrampsLocale) -> bool:
    """True when the locale is Russian.

    The blood-tie rendering is locale-general, but the "former spouse"
    qualifier and the сват/сватья term below are hand-written Russian and only
    applied for this locale.
    """
    return any(code.startswith("ru") for code in (locale.language or []))


def _marriage_dissolved(
    db_handle: DbReadBase, person_a: Person, person_b: Person
) -> bool:
    """True if the family joining person_a and person_b records a Divorce or
    Annulment event (i.e. they are former, not current, spouses).

    Mirrors the core calculator's ex-spouse detection
    (RelationshipCalculator._get_spouse_type), which we cannot reach because
    the in-law path renders via get_single_relationship_string + in_law flags.
    """
    for fam_handle in person_a.get_family_handle_list():
        family = db_handle.get_family_from_handle(fam_handle)
        if family is None:
            continue
        if person_b.handle in (
            family.get_father_handle(),
            family.get_mother_handle(),
        ):
            for eventref in family.get_event_ref_list():
                event = db_handle.get_event_from_handle(eventref.ref)
                if event and event.get_type() in (
                    EventType.DIVORCE,
                    EventType.ANNULMENT,
                ):
                    return True
            return False  # found their family, no dissolution event
    return False


def _ex_prefix(gender: int) -> str:
    """Russian 'former' qualifier agreeing with the kinship-term subject."""
    if gender == Person.MALE:
        return "бывший "
    if gender == Person.FEMALE:
        return "бывшая "
    return "бывш. "


def _spouses(db_handle: DbReadBase, person: Person) -> list[Person]:
    """Return the partner Person objects of ``person`` across all families."""
    spouses = []
    for fam_handle in person.get_family_handle_list():
        family = db_handle.get_family_from_handle(fam_handle)
        if family is None:
            continue
        father = family.get_father_handle()
        mother = family.get_mother_handle()
        if father == person.handle:
            other = mother
        elif mother == person.handle:
            other = father
        else:
            other = None
        if not other:
            continue
        spouse = db_handle.get_person_from_handle(other)
        if spouse is not None:
            spouses.append(spouse)
    return spouses


def _children(db_handle: DbReadBase, person: Person) -> list[Person]:
    """Return the children Person objects of ``person``."""
    kids = []
    for fam_handle in person.get_family_handle_list():
        family = db_handle.get_family_from_handle(fam_handle)
        if family is None:
            continue
        for cref in family.get_child_ref_list():
            child = db_handle.get_person_from_handle(cref.ref)
            if child is not None:
                kids.append(child)
    return kids


def _parents(db_handle: DbReadBase, person: Person) -> list[Person]:
    """Return the parent Person objects of ``person``."""
    pars = []
    for fam_handle in person.get_parent_family_handle_list():
        family = db_handle.get_family_from_handle(fam_handle)
        if family is None:
            continue
        for h in (family.get_father_handle(), family.get_mother_handle()):
            if h:
                par = db_handle.get_person_from_handle(h)
                if par is not None:
                    pars.append(par)
    return pars


def _matchmaker(
    db_handle: DbReadBase, person1: Person, person2: Person
) -> tuple[str, int, int] | None:
    """сват/сватья: person2 is a parent of the spouse of a child of person1.

    Returns ``(string, -1, -1)`` or None. The ``-1`` distances are the
    "no common ancestor" sentinel (this is a one-marriage tie with the
    marriage in the middle of the path, which the calculator cannot express
    via in_law flags). Russian-only; callers gate this on the Russian locale.
    """
    for child in _children(db_handle, person1):
        for child_spouse in _spouses(db_handle, child):
            for parent in _parents(db_handle, child_spouse):
                if parent.handle == person2.handle:
                    ex = _marriage_dissolved(db_handle, child, child_spouse)
                    pre = _ex_prefix(person2.get_gender()) if ex else ""
                    if person2.get_gender() == Person.MALE:
                        return (pre + "сват (отец супруга(и) ребёнка)", -1, -1)
                    if person2.get_gender() == Person.FEMALE:
                        return (pre + "сватья (мать супруга(и) ребёнка)", -1, -1)
                    return (
                        pre + "сват или сватья (родитель супруга(и) ребёнка)",
                        -1,
                        -1,
                    )
    return None


def _render(
    calc: RelationshipCalculator,
    db_handle: DbReadBase,
    blood_a: Person,
    blood_b: Person,
    gender_a: int,
    gender_b: int,
    in_law_a: bool,
    in_law_b: bool,
    locale: GrampsLocale,
    depth: int,
    ex: bool = False,
) -> tuple[str, int, int] | None:
    """Render the blood tie between blood_a/blood_b with in-law flags, or None.

    ``ex`` prefixes the result with a 'former' qualifier (dissolved marriage
    bridge); it agrees with ``gender_b`` (person2, the kinship-term subject).
    The 'former' wording is Russian-only and is skipped for other locales.
    """
    calc.set_depth(min(depth, _MAX_DEPTH))
    data, _msg = calc.get_relationship_distance_new(
        db_handle, blood_a, blood_b, all_dist=True, all_families=True, only_birth=False
    )
    if not data or data[0][0] == -1:
        return None
    data = calc.collapse_relations(data)
    rel = data[0]
    reltocommon_a, reltocommon_b = rel[2], rel[4]
    dist_a, dist_b = len(reltocommon_a), len(reltocommon_b)
    if dist_a == 0 and dist_b == 0:
        return None
    birth = calc.only_birth(reltocommon_a) and calc.only_birth(reltocommon_b)
    if dist_a == 1 and dist_b == 1:
        rel_str = calc.get_sibling_relationship_string(
            calc.get_sibling_type(db_handle, blood_a, blood_b),
            gender_a,
            gender_b,
            in_law_a=in_law_a,
            in_law_b=in_law_b,
        )
    else:
        rel_str = calc.get_single_relationship_string(
            dist_a,
            dist_b,
            gender_a,
            gender_b,
            reltocommon_a,
            reltocommon_b,
            only_birth=birth,
            in_law_a=in_law_a,
            in_law_b=in_law_b,
        )
    if not rel_str:
        return None
    if ex and _is_russian(locale):
        # Where "former" attaches depends on which marriage edge dissolved:
        #  in_law_b — person2 IS the ex-spouse (жена дяди, зять, невестка):
        #             qualify person2's own term ("бывшая жена дяди").
        #  in_law_a — person2 is a BLOOD relative of person1's ex-spouse
        #             (племянник мужа, тёща=мать жены): person2 is NOT "former";
        #             mark the spouse word in the string instead
        #             ("племянник бывшего мужа", "тёща (мать бывшей жены)").
        if in_law_b and not in_law_a:
            rel_str = _ex_prefix(gender_b) + rel_str
        elif in_law_a and not in_law_b:
            marked = rel_str.replace("мужа", "бывшего мужа").replace(
                "жены", "бывшей жены"
            )
            # Fallback for a bare lexicalized term with no spouse word to mark.
            rel_str = marked if marked != rel_str else _ex_prefix(gender_b) + rel_str
    return rel_str, dist_a, dist_b


def inlaw_relationship(
    db_handle: DbReadBase,
    person1: Person,
    person2: Person,
    depth: int,
    locale: GrampsLocale,
) -> tuple[str, int, int] | None:
    """Return (string, dist_a, dist_b) for an in-law tie, else None.

    Describes how ``person2`` is related to ``person1`` through one or two
    marriage edges. Closer ties win: 1-marriage cases (mother-in-law,
    son-in-law, brother-in-law, ...) are tried before 2-marriage cases.

    ``dist_a`` / ``dist_b`` are the blood distances to the marriage bridge,
    not common-ancestor distances; ``-1`` marks a tie with no common ancestor
    (e.g. the сват/сватья case).
    """
    calc = get_relationship_calculator(reinit=True, clocale=locale)
    g1 = person1.get_gender()
    g2 = person2.get_gender()
    sp1 = _spouses(db_handle, person1)
    sp2 = _spouses(db_handle, person2)

    # --- 1 marriage edge ---------------------------------------------------
    # person2 is a blood relative of a spouse of person1 (mother/father-in-law…)
    for spouse in sp1:
        if spouse.handle == person2.handle:
            continue
        ex = _marriage_dissolved(db_handle, person1, spouse)
        result = _render(
            calc, db_handle, spouse, person2, g1, g2, True, False, locale, depth, ex
        )
        if result is not None:
            return result
    # person2 is a spouse of a blood relative of person1 (son/daughter-in-law…)
    for spouse in sp2:
        if spouse.handle == person1.handle:
            continue
        ex = _marriage_dissolved(db_handle, person2, spouse)
        result = _render(
            calc, db_handle, person1, spouse, g1, g2, False, True, locale, depth, ex
        )
        if result is not None:
            return result

    # --- 1 marriage edge, in the middle (сват/сватья, Russian only) --------
    if _is_russian(locale):
        result = _matchmaker(db_handle, person1, person2)
        if result is not None:
            return result

    # --- 2 marriage edges (both ends) -------------------------------------
    # a spouse of person1 is blood-related to a spouse of person2
    for s1 in sp1:
        if s1.handle == person2.handle:
            continue
        for s2 in sp2:
            if s2.handle == person1.handle or s2.handle == s1.handle:
                continue
            result = _render(calc, db_handle, s1, s2, g1, g2, True, True, locale, depth)
            if result is not None:
                return result
    return None
