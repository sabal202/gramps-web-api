#
# Gramps Web API - kinship engine (common ancestors + relatives enumeration)
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

"""Kinship engine for Gramps Web API.

Two public functions:

``common_ancestors(db, person_a, person_b, locale)``
    Returns the closest common ancestor(s) of two people with the
    intermediate path handles and a localized relationship label.

``relatives_of(db, anchor, locale)``
    Returns all relatives of an anchor person grouped by kinship category.
    Blood relatives are derived from a BFS ancestor walk; in-law relatives
    are collected via a bounded candidate scan and the affinal calculator.

Engine only — no Flask resources, no endpoint registration, no schemas.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from gramps.gen.db.base import DbReadBase
from gramps.gen.lib import Person
from gramps.gen.relationship import get_relationship_calculator
from gramps.gen.utils.db import get_birth_or_fallback
from gramps.gen.utils.grampslocale import GrampsLocale

from .inlaw import (
    _children,
    _parents,
    _spouses,
    inlaw_relationship,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Characters that the relationship path builder uses for going *up* the tree.
# Lowercase = birth relation, uppercase = non-birth relation.
_REL_FATHER = "f"
_REL_MOTHER = "m"
_REL_FATHER_NOTBIRTH = "F"
_REL_MOTHER_NOTBIRTH = "M"
# Sibling or family markers may appear in the raw output; we treat them as
# path terminators (no further parent step to follow).
_REL_BIRTH_CHARS = frozenset("fm")
_REL_ANY_PARENT_CHARS = frozenset("fmFM")


def _walk_path_to_ancestor(
    db: DbReadBase,
    start_person: Person,
    path_str: str,
) -> list[str]:
    """Walk *path_str* from *start_person* upward, returning intermediate handles.

    The path string is a sequence of single-character step markers produced by
    ``RelationshipCalculator.__apply_filter``:
      'f' / 'F'  →  go to father (birth / non-birth)
      'm' / 'M'  →  go to mother (birth / non-birth)
      's'        →  sibling marker (no parent step; treated as terminator)
      'a'        →  family marker from collapse_relations (treated as terminator)

    The returned list contains the handles of all persons *strictly between*
    the start person and the common ancestor (both endpoints excluded).

    ``path_str`` from the *raw* (non-collapsed) output ends with the step that
    *arrives at* the common ancestor.  So the intermediate nodes are the
    persons visited after steps [0 .. -2] (i.e. everything except the final
    step which arrives at the ancestor itself, and the start which is excluded
    by definition).
    """
    intermediates: list[str] = []
    current: Person | None = start_person

    for step in path_str:
        if step not in _REL_ANY_PARENT_CHARS:
            # Sibling 's', family 'a', or any future marker: stop climbing.
            break
        if current is None:
            break
        # Find the parent referenced by this step.
        next_handle: str | None = None
        for fam_handle in current.get_parent_family_handle_list():
            family = db.get_family_from_handle(fam_handle)
            if family is None:
                continue
            if step in (_REL_FATHER, _REL_FATHER_NOTBIRTH):
                next_handle = family.get_father_handle()
            else:  # 'm' or 'M'
                next_handle = family.get_mother_handle()
            if next_handle:
                break
        if next_handle is None:
            break
        # Record this node only if it is an *intermediate* (not the last step,
        # which is the common ancestor itself).
        next_person = db.get_person_from_handle(next_handle)
        intermediates.append(next_handle)
        current = next_person

    # The last element we appended is the ancestor; drop it so callers only
    # get the strictly-intermediate persons.
    if intermediates:
        intermediates.pop()

    return intermediates


def _birth_sort_key(db: DbReadBase, handle: str) -> tuple[int, int]:
    """Sort key for a person: (sortval, tiebreak).

    People without a dateable birth event sort last (very large sortval).
    """
    person = db.get_person_from_handle(handle)
    if person is None:
        return (999999999, 0)
    event = get_birth_or_fallback(db, person)
    if event is None:
        return (999999999, 0)
    date = event.get_date_object()
    if date is None or date.is_empty():
        return (999999999, 0)
    return (date.get_sort_value(), 0)


# ---------------------------------------------------------------------------
# Function A — common_ancestors
# ---------------------------------------------------------------------------


def common_ancestors(
    db: DbReadBase,
    person_a: Person,
    person_b: Person,
    locale: GrampsLocale,
) -> dict[str, Any]:
    """Return the closest common ancestor(s) of *person_a* and *person_b*.

    Uses the *raw* (non-collapsed) output of
    ``get_relationship_distance_new`` to preserve per-person ancestor handles
    and build exact intermediate-path lists.

    Returns::

        {
          "relationship": str | None,   # localized a↔b label
          "ancestors": [
            {
              "ancestor_handles": [str, ...],  # 1 normally; 2 for siblings
              "path_a": [str, ...],            # intermediate handles a → ancestor
              "path_b": [str, ...],            # intermediate handles b → ancestor
            },
            ...
          ],
        }

    If the two people are unrelated the dict has ``relationship=None`` and an
    empty ``ancestors`` list.
    """
    calc = get_relationship_calculator(reinit=True, clocale=locale)

    raw_data, _msg = calc.get_relationship_distance_new(
        db,
        person_a,
        person_b,
        all_dist=True,
        all_families=True,
        only_birth=False,
    )

    # raw_data is a list of tuples:
    # (rank, ancestor_handle, path_a_str, fam_a, path_b_str, fam_b)
    if not raw_data or raw_data[0][0] == -1:
        return {"relationship": None, "ancestors": []}

    # Filter to minimum-rank rows only (closest common ancestors).
    min_rank = raw_data[0][0]
    closest = [row for row in raw_data if row[0] == min_rank]

    # Compute the localized relationship label using the collapsed renderer
    # (same pattern as inlaw.py ↔ _render).
    collapsed = calc.collapse_relations(raw_data)
    rel_str: str | None = None
    if collapsed and collapsed[0][0] != -1:
        rel = collapsed[0]
        reltocommon_a = rel[2]  # path string from a (collapsed)
        reltocommon_b = rel[4]  # path string from b (collapsed)
        dist_a = len(reltocommon_a)
        dist_b = len(reltocommon_b)
        ga = person_a.get_gender()
        gb = person_b.get_gender()
        if dist_a == 1 and dist_b == 1:
            rel_str = calc.get_sibling_relationship_string(
                calc.get_sibling_type(db, person_a, person_b),
                ga,
                gb,
            )
        elif dist_a == 0 and dist_b == 0:
            rel_str = None
        else:
            birth = calc.only_birth(reltocommon_a) and calc.only_birth(
                reltocommon_b
            )
            rel_str = calc.get_single_relationship_string(
                dist_a,
                dist_b,
                ga,
                gb,
                reltocommon_a,
                reltocommon_b,
                only_birth=birth,
            )

    # Build the ancestors list from the raw rows.
    ancestors: list[dict[str, Any]] = []
    for row in closest:
        rank, anc_handle, path_a_str, _fam_a, path_b_str, _fam_b = row
        dist_a_raw = len(path_a_str)
        dist_b_raw = len(path_b_str)

        # Sibling branch: dist_a == 1, dist_b == 1, and the path chars are
        # 's' (sibling) rather than 'f'/'m'.  In this case there is no single
        # common ancestor node; instead both parents are the "common ancestors".
        if dist_a_raw == 1 and dist_b_raw == 1 and path_a_str == "s":
            # Collect both parents of person_a as the common ancestor handles.
            ancestor_handles: list[str] = []
            for fam_h in person_a.get_parent_family_handle_list():
                fam = db.get_family_from_handle(fam_h)
                if fam is None:
                    continue
                for ph in (fam.get_father_handle(), fam.get_mother_handle()):
                    if ph and ph not in ancestor_handles:
                        ancestor_handles.append(ph)
            ancestors.append(
                {
                    "ancestor_handles": ancestor_handles,
                    "path_a": [],
                    "path_b": [],
                }
            )
            continue

        # Normal case: anc_handle is the single common ancestor.
        if anc_handle is None:
            continue

        path_a = _walk_path_to_ancestor(db, person_a, path_a_str)
        path_b = _walk_path_to_ancestor(db, person_b, path_b_str)

        ancestors.append(
            {
                "ancestor_handles": [anc_handle],
                "path_a": path_a,
                "path_b": path_b,
            }
        )

    return {"relationship": rel_str or None, "ancestors": ancestors}


# ---------------------------------------------------------------------------
# Function B — relatives_of
# ---------------------------------------------------------------------------

# Mapping of (gen_a, gen_b) → stable category_key.
# gen_a = generations from anchor UP to common ancestor.
# gen_b = generations from the relative DOWN from common ancestor.
# Direct-line persons: gen_a == 0 (descendants) or gen_b == 0 (ancestors).


def _classify(gen_a: int, gen_b: int) -> str:
    """Map (gen_a, gen_b) to a stable category_key string.

    Convention:
      gen_a = steps from anchor up to common ancestor
      gen_b = steps from common ancestor down to relative

    Direct ancestors:  gen_b == 0  (common ancestor IS the relative)
    Direct descendants: gen_a == 0 (anchor IS the common ancestor)
    """
    # Self (should not appear in output but guard anyway)
    if gen_a == 0 and gen_b == 0:
        return "self"

    # Direct ancestors (gen_a = distance from anchor UP to person)
    if gen_b == 0:
        if gen_a == 1:
            return "parents"
        if gen_a == 2:
            return "grandparents"
        if gen_a == 3:
            return "great_grandparents"
        return f"ancestors_{gen_a}"

    # Direct descendants (gen_b = distance from anchor DOWN to person)
    if gen_a == 0:
        if gen_b == 1:
            return "children"
        if gen_b == 2:
            return "grandchildren"
        if gen_b == 3:
            return "great_grandchildren"
        return f"descendants_{gen_b}"

    # Siblings (both are one step from the same parent)
    if gen_a == 1 and gen_b == 1:
        return "siblings"

    # Uncles/aunts: anchor goes up 2+ to common ancestor, relative is 1 below
    if gen_b == 1 and gen_a >= 2:
        if gen_a == 2:
            return "uncle_aunt"
        return f"great_uncle_aunt_{gen_a - 2}" if gen_a > 2 else "uncle_aunt"

    # Nieces/nephews: anchor is 1 below common ancestor, relative goes down 2+
    if gen_a == 1 and gen_b >= 2:
        if gen_b == 2:
            return "niece_nephew"
        return f"great_niece_nephew_{gen_b - 2}" if gen_b > 2 else "niece_nephew"

    # Cousins: both are 2+ steps from common ancestor via collateral line
    # gen_a - 1 == gen_b - 1 → first/second/etc cousin (no removal)
    # Removal = |gen_a - gen_b|, level = min(gen_a, gen_b) - 1
    level = min(gen_a, gen_b) - 1
    removal = abs(gen_a - gen_b)
    if removal == 0:
        if level == 1:
            return "cousins_1"
        if level == 2:
            return "cousins_2"
        return f"cousins_{level}"
    # Removed cousins
    return f"cousins_{level}_removed_{removal}"


def _ancestors_bfs(
    db: DbReadBase, anchor: Person
) -> dict[str, int]:
    """BFS over *anchor*'s ancestors; return {handle: generation}.

    Generation 1 = parents, 2 = grandparents, etc.
    Anchor itself is NOT included.
    """
    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for parent in _parents(db, anchor):
        if parent.handle not in visited:
            visited[parent.handle] = 1
            queue.append((parent.handle, 1))

    while queue:
        handle, gen = queue.popleft()
        person = db.get_person_from_handle(handle)
        if person is None:
            continue
        for parent in _parents(db, person):
            if parent.handle not in visited:
                visited[parent.handle] = gen + 1
                queue.append((parent.handle, gen + 1))

    return visited


def _descendants_bfs(
    db: DbReadBase, root: Person
) -> list[tuple[str, int]]:
    """BFS over *root*'s descendants; return [(handle, generation)].

    Generation 1 = children, 2 = grandchildren, etc.
    Root itself is NOT included.
    """
    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for child in _children(db, root):
        if child.handle not in visited:
            visited[child.handle] = 1
            queue.append((child.handle, 1))

    while queue:
        handle, gen = queue.popleft()
        person = db.get_person_from_handle(handle)
        if person is None:
            continue
        for child in _children(db, person):
            if child.handle not in visited:
                visited[child.handle] = gen + 1
                queue.append((child.handle, gen + 1))

    return list(visited.items())


def _label_blood(
    calc: Any,
    db: DbReadBase,
    anchor: Person,
    person: Person,
    gen_a: int,
    gen_b: int,
    locale: GrampsLocale,
) -> str:
    """Produce a localized blood-relationship label for *person* w.r.t. *anchor*.

    Uses the same renderer pattern as inlaw.py's ``_render`` helper.
    """
    ga = anchor.get_gender()
    gb = person.get_gender()

    if gen_a == 1 and gen_b == 1:
        # Sibling branch — use sibling renderer.
        return calc.get_sibling_relationship_string(
            calc.get_sibling_type(db, anchor, person),
            ga,
            gb,
        )

    # Build minimal synthetic path strings of the correct length (only birth
    # chars).  The actual chars don't matter for the label — only the length
    # and only_birth flag matter for the English/Russian renderers.  We use
    # real path strings when available but for the bulk scanner we can't
    # cheaply recover them; use synthetic ones instead.
    path_a = _REL_MOTHER * gen_a
    path_b = _REL_MOTHER * gen_b
    only_birth = True  # conservative; good enough for labelling

    return calc.get_single_relationship_string(
        gen_a,
        gen_b,
        ga,
        gb,
        path_a,
        path_b,
        only_birth=only_birth,
    )


def relatives_of(
    db: DbReadBase,
    anchor: Person,
    locale: GrampsLocale,
    inlaw_depth: int = 5,
) -> dict[str, Any]:
    """Return all blood and in-law relatives of *anchor* grouped by category.

    Returns::

        {
          "groups": [
            {
              "category_key": str,
              "kind": "blood" | "inlaw",
              "people": [
                {"handle": str, "relationship": str,
                 "gen_a": int, "gen_b": int},
                ...
              ],
            },
            ...
          ],
        }

    Groups are sorted by closeness (gen_a + gen_b ascending); in-law groups
    come after all blood groups.  Within a group people are sorted by birth
    date ascending (undated persons last).
    """
    calc = get_relationship_calculator(reinit=True, clocale=locale)
    anchor_handle = anchor.handle

    # ------------------------------------------------------------------
    # Blood relatives
    # ------------------------------------------------------------------
    # Step 1: collect all ancestors with their generation distance.
    ancestor_gen: dict[str, int] = _ancestors_bfs(db, anchor)

    # Step 2: also include the anchor itself so descendants of the anchor
    # can be found (anchor as root at gen_a = 0).
    # We'll handle the anchor's own descendants separately as a special case
    # (gen_a = 0).

    # blood_map: handle → (gen_a, gen_b) using *closest* common ancestor
    blood_map: dict[str, tuple[int, int]] = {}

    # Collect anchor's direct descendants (gen_a=0).
    for desc_handle, gen_b in _descendants_bfs(db, anchor):
        if desc_handle == anchor_handle:
            continue
        existing = blood_map.get(desc_handle)
        if existing is None or gen_b < existing[1]:
            blood_map[desc_handle] = (0, gen_b)

    # For each ancestor, walk their descendants.
    for anc_handle, gen_a in ancestor_gen.items():
        anc_person = db.get_person_from_handle(anc_handle)
        if anc_person is None:
            continue
        for desc_handle, gen_b in _descendants_bfs(db, anc_person):
            if desc_handle == anchor_handle:
                continue
            total = gen_a + gen_b
            existing = blood_map.get(desc_handle)
            if existing is None or total < existing[0] + existing[1]:
                blood_map[desc_handle] = (gen_a, gen_b)

    # Also include the ancestors themselves (gen_b = 0).
    for anc_handle, gen_a in ancestor_gen.items():
        existing = blood_map.get(anc_handle)
        if existing is None or gen_a < existing[0] + existing[1]:
            blood_map[anc_handle] = (gen_a, 0)

    # Build blood groups.
    blood_groups: dict[str, list[dict[str, Any]]] = {}
    for handle, (gen_a, gen_b) in blood_map.items():
        person = db.get_person_from_handle(handle)
        if person is None:
            continue
        label = _label_blood(calc, db, anchor, person, gen_a, gen_b, locale)
        key = _classify(gen_a, gen_b)
        if key == "self":
            continue
        blood_groups.setdefault(key, []).append(
            {
                "handle": handle,
                "relationship": label,
                "gen_a": gen_a,
                "gen_b": gen_b,
            }
        )

    # ------------------------------------------------------------------
    # In-law relatives — bounded candidate set
    # ------------------------------------------------------------------
    # Candidates:
    #   (a) spouses of blood relatives
    #   (b) blood relatives of anchor's own spouses
    #   (c) parents of spouses of anchor's children (сваты) — covered by (a)+(b)
    blood_handles = set(blood_map.keys())
    inlaw_candidates: set[str] = set()

    # (a) spouses of blood relatives
    for bh in blood_handles:
        bp = db.get_person_from_handle(bh)
        if bp is None:
            continue
        for sp in _spouses(db, bp):
            if (
                sp.handle != anchor_handle
                and sp.handle not in blood_handles
            ):
                inlaw_candidates.add(sp.handle)

    # (b) blood relatives of anchor's spouses
    anchor_spouses = _spouses(db, anchor)
    for sp in anchor_spouses:
        # add the spouse itself (counted as in-law via blood distance 0)
        if sp.handle not in blood_handles:
            inlaw_candidates.add(sp.handle)
        # add blood relatives of the spouse
        sp_ancestors = _ancestors_bfs(db, sp)
        for anc_h, gen_a_sp in sp_ancestors.items():
            if anc_h not in blood_handles and anc_h != anchor_handle:
                inlaw_candidates.add(anc_h)
            anc_p = db.get_person_from_handle(anc_h)
            if anc_p is None:
                continue
            for desc_h, _ in _descendants_bfs(db, anc_p):
                if (
                    desc_h != anchor_handle
                    and desc_h not in blood_handles
                ):
                    inlaw_candidates.add(desc_h)
        # add the spouse itself's ancestors and their descendants
        for desc_h, _ in _descendants_bfs(db, sp):
            if desc_h != anchor_handle and desc_h not in blood_handles:
                inlaw_candidates.add(desc_h)

    # Compute in-law relationships.
    inlaw_groups: dict[str, list[dict[str, Any]]] = {}
    for cand_handle in inlaw_candidates:
        cand = db.get_person_from_handle(cand_handle)
        if cand is None:
            continue
        result = inlaw_relationship(db, anchor, cand, inlaw_depth, locale)
        if result is None:
            continue
        label, dist_a, dist_b = result
        if not label:
            continue
        inlaw_groups.setdefault("inlaw", []).append(
            {
                "handle": cand_handle,
                "relationship": label,
                "gen_a": dist_a if dist_a != -1 else None,
                "gen_b": dist_b if dist_b != -1 else None,
            }
        )

    # ------------------------------------------------------------------
    # Sort within each group by birth date
    # ------------------------------------------------------------------
    for entries in list(blood_groups.values()) + list(inlaw_groups.values()):
        entries.sort(key=lambda e: _birth_sort_key(db, e["handle"]))

    # ------------------------------------------------------------------
    # Assemble output — blood groups sorted by closeness, then in-law
    # ------------------------------------------------------------------
    def _closeness_key(key: str) -> int:
        """Return approximate min(gen_a + gen_b) for a category key for sorting."""
        handles_in_group = blood_groups.get(key, [])
        if not handles_in_group:
            return 999
        return min(
            (e["gen_a"] or 0) + (e["gen_b"] or 0) for e in handles_in_group
        )

    sorted_blood_keys = sorted(blood_groups.keys(), key=_closeness_key)

    groups: list[dict[str, Any]] = []
    for key in sorted_blood_keys:
        groups.append(
            {
                "category_key": key,
                "kind": "blood",
                "people": blood_groups[key],
            }
        )
    for key, people in inlaw_groups.items():
        groups.append(
            {
                "category_key": key,
                "kind": "inlaw",
                "people": people,
            }
        )

    return {"groups": groups}
