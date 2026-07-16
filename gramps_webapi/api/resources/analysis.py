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

"""Whole-tree structural analysis API resources.

Four endpoints built on top of the graph engine in ``graph_analysis.py``
(itself layered on the pure algorithms in ``graph_primitives.py``):

``GET /api/analysis/connectivity/``
    Connected components ("islands") of the whole tree.

``GET /api/analysis/lineages/``
    Whole-tree lineages (brick-wall roots + their descendant closures), or,
    with ``?person=``, the deepest-ancestor lines of just that one person.

``GET /api/analysis/integrity/``
    Data-integrity scans (one-sided back-references, dangling references).

``GET /api/analysis/centrality/``
    Structurally important ("key"/"bridge") people, by centrality metric.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Response
from gramps.gen.db import DbReadBase
from gramps.gen.errors import HandleError
from gramps.gen.lib import Person
from gramps.gen.utils.grampslocale import GrampsLocale

from gramps_webapi.api.people_families_cache import CachePeopleFamiliesProxy

from ..blueprint import api_blueprint
from ..cache import request_cache_decorator
from ..util import abort_with_message, get_db_handle, get_locale_for_language
from . import ProtectedResource
from . import graph_analysis
from .emit import GrampsJSONEncoder
from .schemas import (
    CentralityQueryArgs,
    CentralitySchema,
    ConnectivityQueryArgs,
    ConnectivitySchema,
    IntegrityQueryArgs,
    IntegritySchema,
    LineagesQueryArgs,
    LineagesSchema,
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

    Tries handle first; falls back to gramps_id lookup. (Same logic as
    relatives.py's ``_resolve_person`` — duplicated here because this branch
    predates feat/kinship's relatives.py.)
    """
    try:
        person = db.get_person_from_handle(handle_or_id)
        if person is not None:
            return person
    except HandleError:
        pass
    return db.get_person_from_gramps_id(handle_or_id)


def _person_payload(
    db: DbReadBase,
    person: Person,
    locale: GrampsLocale,
) -> Dict[str, Any]:
    """Build a light person payload (name/sex/birth/death) for a result.

    Trimmed variant of relatives.py's ``_person_payload``: no
    ``media_list``/``relationship`` extras, since analysis results are not
    kinship-labelled and don't need avatar support.
    """
    return get_person_profile_for_object(db, person, args=["age"], locale=locale)


def _person_or_none(
    db: DbReadBase,
    handle: str,
    locale: GrampsLocale,
) -> Optional[Dict[str, Any]]:
    """Hydrate a handle into a person payload, or None if unresolvable."""
    try:
        person = db.get_person_from_handle(handle)
    except HandleError:
        return None
    if person is None:
        return None
    return _person_payload(db, person, locale)


def _people_payload(
    db: DbReadBase,
    handles,
    locale: GrampsLocale,
) -> List[Dict[str, Any]]:
    """Hydrate a list of handles into person payloads, dropping unresolvable ones."""
    out: List[Dict[str, Any]] = []
    for handle in handles:
        payload = _person_or_none(db, handle, locale)
        if payload is not None:
            out.append(payload)
    return out


# ---------------------------------------------------------------------------
# GET /api/analysis/connectivity/
# ---------------------------------------------------------------------------


class ConnectivityResource(ProtectedResource, GrampsJSONEncoder):
    """Connected components ('islands') of the whole tree."""

    @api_blueprint.response(200, ConnectivitySchema())
    @api_blueprint.arguments(ConnectivityQueryArgs, location="query")
    @request_cache_decorator
    def get(self, args: Dict) -> Response:
        """Get connectivity/islands analysis of the whole tree."""
        db_handle = CachePeopleFamiliesProxy(get_db_handle())
        db_handle.cache_people()
        db_handle.cache_families()
        locale = get_locale_for_language(None, default=True)

        result = graph_analysis.connectivity(
            db_handle, include_singletons=args["include_singletons"]
        )

        islands = [
            {
                "size": island["size"],
                "people": _people_payload(db_handle, island["handles"], locale),
            }
            for island in result["islands"]
        ]
        isolated_pairs = [
            {"people": _people_payload(db_handle, pair, locale)}
            for pair in result["isolated_pairs"]
        ]
        orphans = _people_payload(db_handle, result["orphans"], locale)

        return self.response(
            200,
            {
                "person_count": result["person_count"],
                "component_count": result["component_count"],
                "main_component_size": result["main_component_size"],
                "islands": islands,
                "isolated_pairs": isolated_pairs,
                "orphans": orphans,
            },
        )


# ---------------------------------------------------------------------------
# GET /api/analysis/lineages/
# ---------------------------------------------------------------------------


class LineagesResource(ProtectedResource, GrampsJSONEncoder):
    """Whole-tree lineages, or (with ?person=) one person's deepest ancestors."""

    @api_blueprint.response(200, LineagesSchema())
    @api_blueprint.arguments(LineagesQueryArgs, location="query")
    @request_cache_decorator
    def get(self, args: Dict) -> Response:
        """Get whole-tree lineages, or one person's deepest-ancestor lines."""
        db_handle = CachePeopleFamiliesProxy(get_db_handle())
        db_handle.cache_people()
        db_handle.cache_families()
        locale = get_locale_for_language(None, default=True)

        person_param: Optional[str] = args.get("person")
        if person_param:
            return self._get_deepest_ancestors(db_handle, locale, person_param, args)
        return self._get_lineages(db_handle, locale, args)

    def _get_deepest_ancestors(
        self,
        db_handle: DbReadBase,
        locale: GrampsLocale,
        person_param: str,
        args: Dict,
    ) -> Response:
        anchor = _resolve_person(db_handle, person_param)
        if anchor is None:
            abort_with_message(404, f"Person '{person_param}' not found")

        result = graph_analysis.deepest_ancestors(
            db_handle,
            anchor,
            generations=args["generations"],
            birth_only=args["birth_only"],
            top=args["top"],
        )

        by_line = [
            {
                "root": _person_or_none(db_handle, entry["root"], locale),
                "depth": entry["depth"],
                "ancestors": _people_payload(db_handle, entry["ancestors"], locale),
            }
            for entry in result["by_line"]
        ]

        return self.response(
            200,
            {
                "anchor": _person_or_none(db_handle, result["anchor"], locale),
                "max_depth": result["max_depth"],
                "by_line": by_line,
                "furthest": _people_payload(db_handle, result["furthest"], locale),
            },
        )

    def _get_lineages(
        self,
        db_handle: DbReadBase,
        locale: GrampsLocale,
        args: Dict,
    ) -> Response:
        result = graph_analysis.lineages(
            db_handle,
            birth_only=args["birth_only"],
            group_by=args["group_by"],
            min_size=args["min_size"],
        )

        lineages_out: List[Dict[str, Any]] = []
        for entry in result["lineages"][: args["top"]]:
            if "root" in entry:  # group_by == "root"
                lineages_out.append(
                    {
                        "root": _person_or_none(db_handle, entry["root"], locale),
                        "depth": entry["depth"],
                        "size": entry["size"],
                    }
                )
            else:  # group_by == "surname"
                lineages_out.append(
                    {
                        "surname": entry["surname"],
                        "roots": _people_payload(db_handle, entry["roots"], locale),
                        "depth": entry["depth"],
                        "size": entry["size"],
                    }
                )

        return self.response(
            200,
            {
                "max_tree_depth": result["max_tree_depth"],
                "root_count": result["root_count"],
                "lineages": lineages_out,
            },
        )


# ---------------------------------------------------------------------------
# GET /api/analysis/integrity/
# ---------------------------------------------------------------------------


class IntegrityResource(ProtectedResource, GrampsJSONEncoder):
    """Data-integrity scans over the whole tree."""

    @api_blueprint.response(200, IntegritySchema())
    @api_blueprint.arguments(IntegrityQueryArgs, location="query")
    @request_cache_decorator
    def get(self, args: Dict) -> Response:
        """Get data-integrity findings for the whole tree."""
        db_handle = CachePeopleFamiliesProxy(get_db_handle())
        db_handle.cache_people()
        db_handle.cache_families()
        locale = get_locale_for_language(None, default=True)

        result = graph_analysis.integrity(
            db_handle,
            checks=tuple(args["checks"]),
            max_examples=args["max_examples"],
        )

        problems: Dict[str, List[Dict[str, Any]]] = {}
        for check, entries in result["problems"].items():
            hydrated = []
            for entry in entries:
                hydrated.append(
                    {
                        "handle": entry["handle"],
                        "gramps_id": entry["gramps_id"],
                        "detail": entry["detail"],
                        "person": _person_or_none(db_handle, entry["handle"], locale),
                    }
                )
            problems[check] = hydrated

        return self.response(
            200,
            {
                "checks": result["checks"],
                "counts": result["counts"],
                "problems": problems,
            },
        )


# ---------------------------------------------------------------------------
# GET /api/analysis/centrality/
# ---------------------------------------------------------------------------


class CentralityResource(ProtectedResource, GrampsJSONEncoder):
    """Structurally important ('key'/'bridge') people, by centrality metric."""

    @api_blueprint.response(200, CentralitySchema())
    @api_blueprint.arguments(CentralityQueryArgs, location="query")
    @request_cache_decorator
    def get(self, args: Dict) -> Response:
        """Get key/central people in the whole tree, ranked by centrality metric."""
        db_handle = CachePeopleFamiliesProxy(get_db_handle())
        db_handle.cache_people()
        db_handle.cache_families()
        locale = get_locale_for_language(None, default=True)

        result = graph_analysis.centrality(
            db_handle,
            metric=args["metric"],
            top=args["max"],
            max_nodes=args["max_nodes"],
        )

        people = [
            {
                "handle": entry["handle"],
                "score": entry["score"],
                "profile": _person_or_none(db_handle, entry["handle"], locale),
            }
            for entry in result["people"]
        ]

        return self.response(
            200,
            {
                "metric": result["metric"],
                "capped": result["capped"],
                "people": people,
            },
        )
