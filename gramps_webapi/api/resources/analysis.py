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

``GET /api/analysis/graph/``
    Whole-tree graph export (light person nodes + typed edges) for
    client-side graph visualizations.

``GET /api/analysis/semantic-map/``
    2D layout of all people by semantic similarity of their vector
    embeddings (from the semantic search index), for "semantic proximity"
    graph layouts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Response
from gramps.gen.db import DbReadBase
from gramps.gen.errors import HandleError
from gramps.gen.lib import NameOriginType, NameType, Person
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
    GraphSchema,
    IntegrityQueryArgs,
    IntegritySchema,
    LineagesQueryArgs,
    LineagesSchema,
    SemanticMapQueryArgs,
    SemanticMapSchema,
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


# ---------------------------------------------------------------------------
# GET /api/analysis/graph/
# ---------------------------------------------------------------------------


class GraphResource(ProtectedResource, GrampsJSONEncoder):
    """Whole-tree graph export for client-side graph visualizations.

    Returns every visible person as a light node (name, gender, birth/death
    year) plus deduplicated typed edges (spouse↔spouse, parent↔child, as
    indices into the node list). Privacy filtering comes for free: nodes are
    seeded from the proxied ``get_person_handles()``, and edges whose endpoint
    is not in the node set (e.g. a private spouse) are dropped.
    """

    @api_blueprint.response(200, GraphSchema())
    @request_cache_decorator
    def get(self) -> Response:
        """Get the whole tree as a graph of light person nodes and typed edges."""
        db_handle = CachePeopleFamiliesProxy(get_db_handle())
        db_handle.cache_people()
        db_handle.cache_families()

        def _event_year(ref) -> Optional[int]:
            if ref is None:
                return None
            try:
                event = db_handle.get_event_from_handle(ref.ref)
            except HandleError:
                return None
            if event is None:
                return None
            year = event.get_date_object().get_year()
            return year or None

        def _family_surname(name) -> str:
            """Surname without patronymic-origin entries."""
            return " ".join(
                s.get_surname()
                for s in name.get_surname_list()
                if int(s.get_origintype()) != NameOriginType.PATRONYMIC
                and s.get_surname()
            )

        def _patronymic(name) -> str:
            return " ".join(
                s.get_surname()
                for s in name.get_surname_list()
                if int(s.get_origintype()) == NameOriginType.PATRONYMIC
                and s.get_surname()
            )

        def _maiden_surname(person, primary) -> Optional[str]:
            """Birth surname of a woman whose primary name is a married name.

            Mirrors the frontend's getMaidenSurname (used by the maiden-name
            chart toggle): female + primary name typed 'Married Name' + a
            'Birth Name' alternate whose non-patronymic surname differs.
            """
            if person.gender != Person.FEMALE:
                return None
            if int(primary.get_type()) != NameType.MARRIED:
                return None
            for alt in person.get_alternate_names():
                if int(alt.get_type()) == NameType.BIRTH:
                    maiden = _family_surname(alt)
                    if maiden and maiden != _family_surname(primary):
                        return maiden
                    return None
            return None

        def _alt_names(person, primary) -> List[str]:
            """Distinct 'Given Surname' strings of alternate names.

            Lets clients match people by any recorded name variant (birth
            name, AKA, spelling variants), not just the primary one.
            """
            primary_key = f"{primary.get_first_name()} {_family_surname(primary)}"
            out: List[str] = []
            for alt in person.get_alternate_names():
                label = " ".join(
                    part
                    for part in (alt.get_first_name(), _family_surname(alt))
                    if part
                ).strip()
                if label and label != primary_key and label not in out:
                    out.append(label)
            return out

        # tag names are shared via a top-level list; nodes carry indices
        tag_names: List[str] = []
        tag_index: Dict[str, int] = {}

        def _tags(person) -> List[int]:
            out: List[int] = []
            for tag_handle in person.get_tag_list():
                if tag_handle not in tag_index:
                    try:
                        tag = db_handle.get_tag_from_handle(tag_handle)
                    except HandleError:
                        continue
                    if tag is None:
                        continue
                    tag_index[tag_handle] = len(tag_names)
                    tag_names.append(tag.get_name())
                out.append(tag_index[tag_handle])
            return out

        def _citation_count(person) -> int:
            """Citations on the person plus on their events."""
            count = len(person.get_citation_list())
            for event_ref in person.get_event_ref_list():
                try:
                    event = db_handle.get_event_from_handle(event_ref.ref)
                except HandleError:
                    continue
                if event is not None:
                    count += len(event.get_citation_list())
            return count

        people: List[Dict[str, Any]] = []
        index: Dict[str, int] = {}
        for handle in db_handle.get_person_handles():
            person = db_handle.get_person_from_handle(handle)
            if person is None:
                continue
            name = person.get_primary_name()
            index[handle] = len(people)
            people.append(
                {
                    "handle": handle,
                    "gramps_id": person.gramps_id,
                    "given_name": name.get_first_name(),
                    "surname": name.get_surname(),
                    "family_surname": _family_surname(name),
                    "patronymic": _patronymic(name),
                    "maiden_surname": _maiden_surname(person, name),
                    "alt_names": _alt_names(person, name),
                    "gender": person.gender,
                    "birth_year": _event_year(person.get_birth_ref()),
                    "death_year": _event_year(person.get_death_ref()),
                    "tags": _tags(person),
                    "citations": _citation_count(person),
                }
            )

        links: List[Dict[str, Any]] = []
        seen: set = set()

        def _add_link(a: Optional[str], b: Optional[str], type_: str) -> None:
            source = index.get(a) if a else None
            target = index.get(b) if b else None
            if source is None or target is None or source == target:
                return
            key = (source, target, type_)
            if key in seen:
                return
            seen.add(key)
            links.append({"source": source, "target": target, "type": type_})

        for family in db_handle.iter_families():
            father = family.get_father_handle()
            mother = family.get_mother_handle()
            _add_link(father, mother, "spouse")
            for cref in family.get_child_ref_list():
                child = cref.ref
                if not child:
                    continue
                _add_link(father, child, "child")
                _add_link(mother, child, "child")

        return self.response(
            200, {"people": people, "links": links, "tags": tag_names}
        )


# ---------------------------------------------------------------------------
# GET /api/analysis/semantic-map/
# ---------------------------------------------------------------------------


class SemanticMapResource(ProtectedResource, GrampsJSONEncoder):
    """2D semantic-similarity layout of all people.

    Reads the per-person vector embeddings from the semantic search index
    (the same ones powering semantic search / AI chat retrieval) and reduces
    them to two dimensions — UMAP when available, PCA otherwise. Coordinates
    are normalized to roughly [-1, 1] per axis. Privacy: users without the
    view-private permission get the public-only embedding collection.
    """

    @api_blueprint.response(200, SemanticMapSchema())
    @api_blueprint.arguments(SemanticMapQueryArgs, location="query")
    @request_cache_decorator
    def get(self, args: Dict) -> Response:
        """Get a 2D semantic-similarity map of all people."""
        import numpy as np

        from ..auth import has_permissions
        from ..search import get_semantic_search_indexer
        from ...auth.const import PERM_VIEW_PRIVATE
        from ..util import get_tree_from_jwt

        tree = get_tree_from_jwt()
        try:
            indexer = get_semantic_search_indexer(tree)
        except ValueError:
            abort_with_message(501, "Semantic search index is not configured")
        collection = (
            indexer.index
            if has_permissions({PERM_VIEW_PRIVATE})
            else indexer.index_public
        )

        handles: List[str] = []
        vectors: List[Any] = []
        import json as _json

        with collection.conn() as conn:
            placeholder = collection.PLACEHOLDER
            cursor = conn.execute(
                "SELECT metadata, embedding FROM documents "
                f"WHERE name = {placeholder} AND embedding IS NOT NULL "
                "AND id LIKE 'person_%'",
                (collection.name,),
            )
            for metadata, embedding in cursor.fetchall():
                try:
                    meta = (
                        _json.loads(metadata)
                        if isinstance(metadata, str)
                        else metadata
                    )
                except (TypeError, ValueError):
                    continue
                if not meta or meta.get("type") != "person":
                    continue
                handle = meta.get("handle")
                if not handle:
                    continue
                if isinstance(embedding, (bytes, memoryview)):
                    vector = np.frombuffer(bytes(embedding), dtype=np.float32)
                else:
                    vector = np.asarray(embedding, dtype=np.float32)
                handles.append(handle)
                vectors.append(vector)

        if len(handles) < 3:
            return self.response(
                200, {"method": "none", "people": []}
            )

        matrix = np.vstack(vectors).astype(np.float32)
        method = args["method"]
        coords: Any = None
        if method in ("auto", "umap"):
            try:
                import umap  # type: ignore

                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )
                coords = reducer.fit_transform(matrix)
                method = "umap"
            except ImportError:
                if method == "umap":
                    abort_with_message(501, "umap-learn is not installed")
                method = "pca"
        if coords is None:
            centered = matrix - matrix.mean(axis=0)
            cov = centered.T @ centered
            _eigvals, eigvecs = np.linalg.eigh(cov)
            coords = centered @ eigvecs[:, -2:]
            method = "pca"

        # robust per-axis normalization to ~[-1, 1] (98th percentile of |v|)
        coords = np.asarray(coords, dtype=np.float64)
        coords = coords - coords.mean(axis=0)
        scale = np.percentile(np.abs(coords), 98, axis=0)
        scale[scale == 0] = 1.0
        coords = np.clip(coords / scale, -1.5, 1.5)

        people = [
            {
                "handle": handle,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            }
            for i, handle in enumerate(handles)
        ]
        return self.response(200, {"method": method, "people": people})
