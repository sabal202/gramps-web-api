#
# Gramps Web API - Immich -> Gramps photo import
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

"""Immich <-> Gramps mapping storage and name-match suggestions.

Per the design doc, all three mappings are stored as Gramps **Attributes**
(no side sqlite/json):

============================  ==================  ==================
Mapping                       Stored as           On object
============================  ==================  ==================
Immich person -> Gramps person ``ImmichPersonId``   Person
Immich asset -> Gramps media   ``ImmichAssetId``    Media
Immich album -> Gramps target  ``ImmichAlbumId``    the target object
============================  ==================  ==================

Module layout - deliberately split into two halves:

1. **Pure name-match helpers** (``score_name_match``, ``suggest_person_matches``)
   have zero gramps/gi imports at module scope and are unit-tested standalone
   by loading this file directly via ``importlib.util.spec_from_file_location``
   (see ``tests_immich_import/test_mapping_namematch.py``) - that bypasses the
   ``gramps_webapi`` package ``__init__`` chain entirely, which pulls in ``gi``
   as soon as it is dotted-imported.
2. **DB-touching helpers** (attribute read/write, person/target resolution)
   import gramps lazily *inside* each function body. That keeps this file
   importable without gi while still being usable normally (with real relative
   imports resolved by the package machinery) once the Flask app actually
   calls these functions inside the gi-container.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Attribute type constants (shared with gramps_writer.py)
# ---------------------------------------------------------------------------

ATTR_IMMICH_PERSON_ID = "ImmichPersonId"
ATTR_IMMICH_ASSET_ID = "ImmichAssetId"
ATTR_IMMICH_ALBUM_ID = "ImmichAlbumId"

# Primary object class names a Flow-B target may be (matches
# gramps_webapi.const.GRAMPS_NAMESPACES values, duplicated here as a plain
# tuple so this module doesn't need to import gramps_webapi.const at top
# level - that import is fine at runtime, it's just done lazily below so this
# whole file stays gi-free to import).
TARGETABLE_CLASSES = (
    "Person",
    "Family",
    "Event",
    "Place",
    "Citation",
    "Source",
    "Repository",
    "Note",
    "Media",
)


# ---------------------------------------------------------------------------
# Pure name-match helpers (no gramps/gi import - unit tested standalone)
# ---------------------------------------------------------------------------


def _normalize_name(text: str) -> str:
    """Casefold + accent-fold + collapse whitespace for fuzzy comparison."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_name_match(
    immich_name: str, name_given: Optional[str], name_surname: Optional[str]
) -> float:
    """Return a 0..1 similarity score between an Immich person name and a
    Gramps candidate's given/surname.

    Combines whole-string fuzzy similarity (``difflib.SequenceMatcher``,
    tried in both "given surname" and "surname given" order, since Immich
    person names have no given/surname split) with a token-overlap score,
    and returns the higher of the two. Token overlap catches cases where
    word order or a missing patronymic would otherwise tank the fuzzy ratio
    (e.g. "Ivan Petrov" vs "Petrov Ivan Sergeevich").
    """
    norm_immich = _normalize_name(immich_name)
    if not norm_immich:
        return 0.0

    full_given_surname = _normalize_name(f"{name_given or ''} {name_surname or ''}")
    full_surname_given = _normalize_name(f"{name_surname or ''} {name_given or ''}")

    score = max(
        _similarity(norm_immich, full_given_surname),
        _similarity(norm_immich, full_surname_given),
    )

    immich_tokens = set(norm_immich.split())
    candidate_tokens = set(full_given_surname.split())
    if immich_tokens and candidate_tokens:
        overlap = len(immich_tokens & candidate_tokens) / len(
            immich_tokens | candidate_tokens
        )
        score = max(score, overlap)

    return round(score, 4)


def suggest_person_matches(
    immich_name: str,
    candidates: Iterable[dict[str, Any]],
    *,
    limit: int = 5,
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """Rank Gramps person candidates by name similarity to an Immich person.

    ``candidates`` is an iterable of dicts with at least ``handle``,
    ``name_given``, ``name_surname`` keys (extra keys are passed through
    unchanged). Returns the top ``limit`` candidates scoring >= ``min_score``,
    each with a ``score`` key added, sorted by score descending (stable for
    ties - keeps the input order among equal scores).

    Never used to auto-link silently (per design "Open risks" - name-match
    quality is mitigated by curator confirmation); this only produces
    *suggestions* for the UI to present.
    """
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        score = score_name_match(
            immich_name,
            candidate.get("name_given"),
            candidate.get("name_surname"),
        )
        if score >= min_score:
            scored.append({**candidate, "score": score})

    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# DB-touching helpers (lazy gramps imports inside function bodies only)
# ---------------------------------------------------------------------------


def get_attribute_value(obj: Any, attr_type: str) -> Optional[str]:
    """Return the value of the first ``Attribute`` of type ``attr_type`` on
    ``obj``, or ``None``. Works on any object with an attribute list
    (Person, Media, MediaRef, ...) via duck typing - no gramps import needed
    to *read*, only to construct a new ``Attribute`` (see
    ``set_attribute_value``).
    """
    for attr in obj.get_attribute_list():
        if str(attr.get_type()) == attr_type:
            return attr.get_value()
    return None


def set_attribute_value(obj: Any, attr_type: str, value: str) -> None:
    """Set (or create) the ``Attribute`` of type ``attr_type`` on ``obj``.

    Caller is responsible for committing ``obj`` afterwards (via
    ``gramps_writer``'s ``DbTxn``/``update_object`` pattern).
    """
    for attr in obj.get_attribute_list():
        if str(attr.get_type()) == attr_type:
            attr.set_value(value)
            return
    from gramps.gen.lib import Attribute  # lazy: only needed to create new

    attr = Attribute()
    attr.set_type(attr_type)
    attr.set_value(value)
    obj.add_attribute(attr)


def find_media_by_immich_asset_id(db_handle: Any, immich_asset_id: str) -> Optional[str]:
    """Return the Gramps Media handle already tagged with this
    ``ImmichAssetId``, or ``None`` if this asset has never been imported.
    O(n) full-tree scan (same tradeoff as ``find_person_by_immich_id``).
    """
    for media in db_handle.iter_media():
        if get_attribute_value(media, ATTR_IMMICH_ASSET_ID) == immich_asset_id:
            return media.handle
    return None


def match_external_assets_to_media(
    db_handle: Any, assets: Iterable[dict[str, Any]]
) -> list[tuple[str, dict[str, Any]]]:
    """Match Immich external-library assets to existing Gramps Media objects.

    Primary (only) match key: basename of the asset's original path vs.
    basename of the Media object's path. The external library is a
    read-only mount of the *same* directory tree Gramps media lives in
    (``tank/gramps/media``), so filenames are expected to be identical
    (design doc "Open risks": "assumes external-lib mirrors exact Gramps
    media files (true today) - verified at match time, mismatches surfaced
    not guessed"). Checksum comparison was considered but Immich's asset
    checksum is a base64 sha1 while Gramps' is a hex sha256 - not directly
    comparable without re-hashing the file, so basename is the practical
    signal for v1.
    """
    import os

    assets_by_basename: dict[str, dict[str, Any]] = {}
    for asset in assets:
        original_path = asset.get("originalPath") or ""
        if not original_path:
            continue
        assets_by_basename.setdefault(os.path.basename(original_path), asset)

    matches: list[tuple[str, dict[str, Any]]] = []
    for media in db_handle.iter_media():
        basename = os.path.basename(media.path or "")
        asset = assets_by_basename.get(basename)
        if asset is not None:
            matches.append((media.handle, asset))
    return matches


def find_person_by_immich_id(db_handle: Any, immich_person_id: str) -> Optional[str]:
    """Return the Gramps handle of the person mapped to this Immich person id,
    or ``None`` if unmapped. O(n) full-tree scan - acceptable for the tree
    sizes this deploys against (hundreds of people); revisit with an index
    if that stops being true.
    """
    for person in db_handle.iter_people():
        if get_attribute_value(person, ATTR_IMMICH_PERSON_ID) == immich_person_id:
            return person.handle
    return None


def get_immich_person_id_for_handle(
    db_handle: Any, gramps_handle: str
) -> Optional[str]:
    """Return the ``ImmichPersonId`` attribute value for a Gramps person
    handle, or ``None`` if the person doesn't exist or isn't mapped.
    """
    from gramps.gen.errors import HandleError

    try:
        person = db_handle.get_person_from_handle(gramps_handle)
    except HandleError:
        return None
    return get_attribute_value(person, ATTR_IMMICH_PERSON_ID)


def set_immich_person_mapping(
    db_handle: Any, trans: Any, gramps_handle: str, immich_person_id: str
) -> None:
    """Upsert the ``ImmichPersonId`` attribute on the given Gramps person and
    commit the change within the caller's ``DbTxn``.
    """
    from gramps.gen.errors import HandleError

    from ..util import update_object  # gramps_webapi.api.resources.util

    try:
        person = db_handle.get_person_from_handle(gramps_handle)
    except HandleError as exc:
        raise ValueError(f"Person handle {gramps_handle!r} not found") from exc
    set_attribute_value(person, ATTR_IMMICH_PERSON_ID, immich_person_id)
    update_object(db_handle, person, trans)


def resolve_target(db_handle: Any, gramps_id: str) -> Optional[tuple[str, str]]:
    """Resolve a Flow-B import target given only its Gramps ID string.

    Tries every primary object class the frozen endpoint contract allows as
    an import target (event/family/person/place/... - see
    ``TARGETABLE_CLASSES``) and returns ``(class_name, handle)`` for the
    first match, or ``None``. Gramps IDs are unique per *class*, not
    globally, but in practice a curator picks one object by its ID from a
    single search result, so the first hit is the intended one.
    """
    from gramps.gen.errors import HandleError

    for class_name in TARGETABLE_CLASSES:
        method = db_handle.method("get_%s_from_gramps_id", class_name)
        if method is None:
            continue
        try:
            obj = method(gramps_id)
        except HandleError:
            obj = None
        if obj is not None:
            return class_name, obj.handle
    return None


def find_target_by_immich_album(
    db_handle: Any, immich_album_id: str
) -> Optional[tuple[str, str]]:
    """Return ``(class_name, handle)`` of the tree object carrying this
    ``ImmichAlbumId``, or ``None`` if the album has never been mapped.

    O(n) full-tree scan across every targetable class (no reverse index -
    matches the same tradeoff as ``find_person_by_immich_id``). An album
    maps to exactly one target; a target may carry several ``ImmichAlbumId``
    attribute values (one per album), so this returns the first object whose
    attribute list contains a value equal to ``immich_album_id``.
    """
    from ....const import GRAMPS_OBJECT_PLURAL  # gramps_webapi.const

    for class_name in TARGETABLE_CLASSES:
        plural = GRAMPS_OBJECT_PLURAL.get(class_name)
        if plural is None:
            continue
        iter_func = getattr(db_handle, f"iter_{plural}", None)
        if iter_func is None:
            continue
        for obj in iter_func():
            if get_attribute_value(obj, ATTR_IMMICH_ALBUM_ID) == immich_album_id:
                return class_name, obj.handle
    return None


def set_album_target_mapping(
    db_handle: Any,
    trans: Any,
    class_name: str,
    handle: str,
    immich_album_id: str,
) -> None:
    """Add an ``ImmichAlbumId`` attribute value to the target object and
    commit within the caller's ``DbTxn``.

    An album maps to exactly one target, but a target may accumulate several
    ``ImmichAlbumId`` values over time (several albums importing into the
    same event/person/...), so this always *adds* a fresh attribute rather
    than overwriting an existing one - unlike ``set_attribute_value``, which
    is single-valued and used for the 1:1 person/asset mappings.
    """
    from gramps.gen.errors import HandleError
    from gramps.gen.lib import Attribute

    from ..util import update_object  # gramps_webapi.api.resources.util

    get_method = db_handle.method("get_%s_from_handle", class_name)
    if get_method is None:
        raise ValueError(f"Unknown target class {class_name!r}")
    try:
        obj = get_method(handle)
    except HandleError as exc:
        raise ValueError(f"{class_name} handle {handle!r} not found") from exc

    # Avoid duplicate attribute values if the same album is remapped twice.
    existing_values = {
        attr.get_value()
        for attr in obj.get_attribute_list()
        if str(attr.get_type()) == ATTR_IMMICH_ALBUM_ID
    }
    if immich_album_id not in existing_values:
        attr = Attribute()
        attr.set_type(ATTR_IMMICH_ALBUM_ID)
        attr.set_value(immich_album_id)
        obj.add_attribute(attr)
        update_object(db_handle, obj, trans)
