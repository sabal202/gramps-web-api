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

"""Relatives and Common-Ancestors API resources.

Two endpoints built on top of the kinship engine in ``kinship.py``:

``GET /api/relatives/``
    All relatives of the home person (or an optional ``?handle=`` override),
    grouped by kinship category, with person profiles.

``GET /api/people/<handle>/common-ancestors``
    Closest common ancestor(s) between the given person and the home person
    (or an optional ``?to=`` override), with person profiles and relationship label.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Response
from gramps.gen.db import DbReadBase
from gramps.gen.errors import HandleError
from gramps.gen.lib import Person
from gramps.gen.utils.grampslocale import GrampsLocale

from gramps_webapi.api.people_families_cache import CachePeopleFamiliesProxy

from ...types import Handle
from ..blueprint import api_blueprint
from ..util import abort_with_message, get_db_handle, get_locale_for_language
from . import ProtectedResource
from .emit import GrampsJSONEncoder
from .kinship import common_ancestors, relatives_of
from .schemas import (
    CommonAncestorsQueryArgs,
    CommonAncestorsSchema,
    RelativesQueryArgs,
    RelativesSchema,
)
from .util import get_person_profile_for_object

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_person(
    db: DbReadBase,
    handle_or_id: str,
) -> Optional[Person]:
    """Resolve a handle-or-gramps_id string to a Person, or return None.

    Tries handle first; falls back to gramps_id lookup.
    """
    try:
        person = db.get_person_from_handle(handle_or_id)
        if person is not None:
            return person
    except HandleError:
        pass
    # Try as gramps_id
    return db.get_person_from_gramps_id(handle_or_id)


def _person_payload(
    db: DbReadBase,
    person: Person,
    locale: GrampsLocale,
    relationship: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a person payload dict with profile fields + media_list.

    Includes: handle, gramps_id, name_given, name_surname, sex, birth, death,
    name_display, name_suffix, media_list.  The ``age`` arg populates
    ``death.age`` for deceased persons (matching the rest of the app's person
    cards).  If ``relationship`` is not None it is added under that key.
    """
    profile = get_person_profile_for_object(
        db,
        person,
        args=["age"],
        locale=locale,
    )
    # Attach media_list for avatar support (not included in the light profile).
    profile["media_list"] = [
        {"ref": r.ref, "rect": r.rect} for r in person.get_media_list()
    ]
    if relationship is not None:
        profile["relationship"] = relationship
    return profile


# ---------------------------------------------------------------------------
# GET /api/relatives/
# ---------------------------------------------------------------------------


class RelativesResource(ProtectedResource, GrampsJSONEncoder):
    """All relatives of the home person, grouped by kinship category."""

    @api_blueprint.response(200, RelativesSchema())
    @api_blueprint.arguments(RelativesQueryArgs, location="query")
    def get(self, args: Dict) -> Response:
        """Get relatives of the home person (``?handle=`` override), grouped."""
        db_handle = CachePeopleFamiliesProxy(get_db_handle())

        # Resolve anchor
        handle_param: Optional[str] = args.get("handle")
        if handle_param:
            anchor = _resolve_person(db_handle, handle_param)
            if anchor is None:
                abort_with_message(404, f"Person '{handle_param}' not found")
        else:
            anchor = db_handle.get_default_person()
            if anchor is None:
                abort_with_message(
                    400,
                    "No home person set and no ?handle= provided",
                )

        db_handle.cache_people()
        db_handle.cache_families()

        locale = get_locale_for_language(args.get("locale"), default=True)

        # Run the kinship engine
        result = relatives_of(db_handle, anchor, locale)

        # Build anchor payload (no relationship key)
        anchor_payload = _person_payload(db_handle, anchor, locale)

        # Build groups with full person profiles
        groups: List[Dict[str, Any]] = []
        for group in result["groups"]:
            people_out: List[Dict[str, Any]] = []
            for entry in group["people"]:
                try:
                    person = db_handle.get_person_from_handle(entry["handle"])
                except HandleError:
                    continue
                if person is None:
                    continue
                payload = _person_payload(
                    db_handle,
                    person,
                    locale,
                    relationship=entry["relationship"],
                )
                # Propagate per-person kind (blood / inlaw) to the payload.
                payload["kind"] = entry.get("kind", "blood")
                people_out.append(payload)
            groups.append(
                {
                    "category_key": group["category_key"],
                    "count": len(people_out),
                    "people": people_out,
                }
            )

        return self.response(
            200,
            {
                "anchor": anchor_payload,
                "groups": groups,
            },
        )


# ---------------------------------------------------------------------------
# GET /api/people/<handle>/common-ancestors
# ---------------------------------------------------------------------------


class CommonAncestorsResource(ProtectedResource, GrampsJSONEncoder):
    """Closest common ancestor(s) between a person and the home person."""

    @api_blueprint.response(200, CommonAncestorsSchema())
    @api_blueprint.arguments(CommonAncestorsQueryArgs, location="query")
    def get(self, args: Dict, handle: Handle) -> Response:
        """Get common ancestors of ``handle`` and home person (``?to=`` overrides)."""
        db_handle = CachePeopleFamiliesProxy(get_db_handle())

        # Resolve subject (path param)
        try:
            subject = db_handle.get_person_from_handle(handle)
        except HandleError:
            subject = None
        if subject is None:
            abort_with_message(404, f"Person '{handle}' not found")

        # Resolve other (home person or ?to= override)
        to_param: Optional[str] = args.get("to")
        if to_param:
            other = _resolve_person(db_handle, to_param)
            if other is None:
                abort_with_message(404, f"Person '{to_param}' not found")
        else:
            other = db_handle.get_default_person()
            # No home person is not an error — return null/empty result (200)
            if other is None:
                return self.response(
                    200,
                    {"relationship": None, "ancestors": []},
                )

        db_handle.cache_people()
        db_handle.cache_families()

        locale = get_locale_for_language(args.get("locale"), default=True)

        # Run the kinship engine
        result = common_ancestors(db_handle, subject, other, locale)

        # Build ancestors list with person profiles
        ancestors_out: List[Dict[str, Any]] = []
        for entry in result["ancestors"]:
            # Common ancestor profiles
            anc_profiles: List[Dict[str, Any]] = []
            for h in entry["ancestor_handles"]:
                try:
                    p = db_handle.get_person_from_handle(h)
                except HandleError:
                    continue
                if p is not None:
                    anc_profiles.append(_person_payload(db_handle, p, locale))

            # Intermediate path profiles
            path_a_profiles: List[Dict[str, Any]] = []
            for h in entry["path_a"]:
                try:
                    p = db_handle.get_person_from_handle(h)
                except HandleError:
                    continue
                if p is not None:
                    path_a_profiles.append(_person_payload(db_handle, p, locale))

            path_b_profiles: List[Dict[str, Any]] = []
            for h in entry["path_b"]:
                try:
                    p = db_handle.get_person_from_handle(h)
                except HandleError:
                    continue
                if p is not None:
                    path_b_profiles.append(_person_payload(db_handle, p, locale))

            ancestors_out.append(
                {
                    "common_ancestors": anc_profiles,
                    "path_a": path_a_profiles,
                    "path_b": path_b_profiles,
                }
            )

        return self.response(
            200,
            {
                "relationship": result["relationship"],
                "ancestors": ancestors_out,
            },
        )
