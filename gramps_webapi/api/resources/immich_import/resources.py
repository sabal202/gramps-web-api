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

"""REST endpoints for the Immich import cluster.

⚠️ gi-dependent - cannot be imported or run on Windows. Only
``python -m py_compile`` was used to catch syntax errors locally; must be
exercised for real in the gi-container on the NAS.

All endpoints gated ``PERM_EDIT_OBJ`` (Editor+), per the design doc's "Error
handling" section, matching how every other tree-mutating cluster in this
codebase is gated (e.g. ``object_history.py``, ``inlaw.py``).
"""

from __future__ import annotations

import mimetypes
from typing import Any, Dict

import requests
from flask import Response
from gramps.gen.db import DbTxn

from ....auth.const import PERM_EDIT_OBJ
from ...auth import require_permissions
from ...blueprint import api_blueprint
from ...media import get_media_handler, update_usage_media
from ...util import abort_with_message, get_db_handle, get_tree_from_jwt
from .. import ProtectedResource
from ..emit import GrampsJSONEncoder
from ..schemas import (
    ImmichAlbumCommitArgs,
    ImmichAlbumCommitResultSchema,
    ImmichAlbumPreviewQueryArgs,
    ImmichAlbumPreviewSchema,
    ImmichAlbumSchema,
    ImmichExistingCommitArgs,
    ImmichExistingCommitResultSchema,
    ImmichExistingPreviewSchema,
    ImmichMappingArgs,
    ImmichPeopleQueryArgs,
    ImmichPersonSchema,
)
from .gramps_writer import (
    add_person_region,
    attach_media_to_target,
    fetch_and_validate_asset,
    find_or_create_media,
    person_has_region_on_media,
)
from .immich_client import ImmichApiError, ImmichClient
from .mapping import (
    find_media_by_immich_asset_id,
    find_person_by_immich_id,
    find_target_by_immich_album,
    match_external_assets_to_media,
    resolve_target,
    set_album_target_mapping,
    set_immich_person_mapping,
)
from .regions import face_bbox_to_gramps_rect, portrait_region_from_face

# Errors that mean "the Immich API is unreachable/misbehaving" rather than a
# bug in our request - surfaced as 502 so the tab can degrade gracefully
# (design doc "Error handling": "Immich API down -> tab degrades with a
# message; does not break Gramps").
_IMMICH_UNAVAILABLE_ERRORS = (ImmichApiError, requests.RequestException)


def _guess_mime(asset_metadata: Dict[str, Any]) -> str:
    """Best-effort MIME type for a downloaded asset.

    The frozen commit request body only carries ``immichAssetId``/``attach``/
    ``regions`` (no MIME type from the frontend), so this re-derives it from
    the asset metadata fetched server-side: prefer Immich's own
    ``originalMimeType`` field, falling back to guessing from
    ``originalFileName``'s extension.
    """
    mime = asset_metadata.get("originalMimeType")
    if mime:
        return mime
    filename = asset_metadata.get("originalFileName") or ""
    guessed, _encoding = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _target_gramps_id(db_handle: Any, class_name: str, handle: str) -> str | None:
    """Return the Gramps ID string for a resolved (class_name, handle) pair."""
    get_method = db_handle.method("get_%s_from_handle", class_name)
    if get_method is None:
        return None
    obj = get_method(handle)
    return obj.gramps_id if obj is not None else None


def _face_to_person_match(face: Dict[str, Any], db_handle: Any) -> Dict[str, Any] | None:
    """Convert one ``GET /faces`` entry into an ``ImmichPersonMatchSchema``
    dict, or ``None`` if the face is missing the image dimensions needed to
    compute a percent rect (should not normally happen, defensive only).
    """
    width = face.get("imageWidth")
    height = face.get("imageHeight")
    if not width or not height:
        return None

    x1 = face["boundingBoxX1"]
    y1 = face["boundingBoxY1"]
    x2 = face["boundingBoxX2"]
    y2 = face["boundingBoxY2"]

    bbox_pct = face_bbox_to_gramps_rect(x1, y1, x2, y2, width, height)
    proposed = portrait_region_from_face(x1, y1, x2, y2, width, height)

    person = face.get("person")
    immich_person_id = person.get("id") if person else None
    name = person.get("name") if person else None
    gramps_handle = (
        find_person_by_immich_id(db_handle, immich_person_id)
        if immich_person_id
        else None
    )

    return {
        "immichPersonId": immich_person_id,
        "name": name,
        "bbox": list(bbox_pct),
        "grampsHandle": gramps_handle,
        "proposedRect": list(proposed),
    }


# ---------------------------------------------------------------------------
# GET /api/immich/albums/
# ---------------------------------------------------------------------------


class ImmichAlbumsResource(ProtectedResource, GrampsJSONEncoder):
    """List the archive account's Immich albums."""

    @api_blueprint.response(200, ImmichAlbumSchema(many=True))
    def get(self) -> Response:
        """List Immich albums with their remembered Gramps target, if any."""
        require_permissions([PERM_EDIT_OBJ])
        db_handle = get_db_handle()
        client = ImmichClient()

        try:
            raw_albums = client.list_albums()
        except _IMMICH_UNAVAILABLE_ERRORS as exc:
            abort_with_message(502, f"Immich API unavailable: {exc}")

        albums = []
        for album in raw_albums:
            mapped_gramps_id = None
            target = find_target_by_immich_album(db_handle, album["id"])
            if target is not None:
                class_name, handle = target
                mapped_gramps_id = _target_gramps_id(db_handle, class_name, handle)
            albums.append(
                {
                    "id": album["id"],
                    "name": album.get("albumName", ""),
                    "assetCount": album.get("assetCount", 0),
                    "mappedTargetGrampsId": mapped_gramps_id,
                }
            )
        return self.response(200, albums, total_items=len(albums))


# ---------------------------------------------------------------------------
# GET /api/immich/albums/<album_id>/preview
# ---------------------------------------------------------------------------


class ImmichAlbumPreviewResource(ProtectedResource, GrampsJSONEncoder):
    """Flow B preview: album assets + proposed attachment + face regions."""

    @api_blueprint.response(200, ImmichAlbumPreviewSchema())
    @api_blueprint.arguments(ImmichAlbumPreviewQueryArgs, location="query")
    def get(self, args: Dict, album_id: str) -> Response:
        """Preview an album's assets against the chosen target object."""
        require_permissions([PERM_EDIT_OBJ])
        db_handle = get_db_handle()

        # Only used to validate the target exists before spending API calls
        # on the album; the preview response itself doesn't need the
        # resolved (class_name, handle) - that's only needed at commit time.
        if resolve_target(db_handle, args["target"]) is None:
            abort_with_message(404, f"Target object {args['target']!r} not found")

        client = ImmichClient()
        try:
            assets = list(client.album_assets(album_id))
        except _IMMICH_UNAVAILABLE_ERRORS as exc:
            abort_with_message(502, f"Immich API unavailable: {exc}")

        items = []
        for asset in assets:
            asset_id = asset["id"]
            already_imported = (
                find_media_by_immich_asset_id(db_handle, asset_id) is not None
            )
            try:
                faces = client.asset_faces(asset_id)
            except _IMMICH_UNAVAILABLE_ERRORS:
                faces = []

            people = [
                match
                for face in faces
                if (match := _face_to_person_match(face, db_handle)) is not None
            ]

            items.append(
                {
                    "immichAssetId": asset_id,
                    "thumbnailUrl": (
                        f"{client.base_url}/api/assets/{asset_id}/thumbnail"
                    ),
                    "alreadyImported": already_imported,
                    "people": people,
                }
            )

        return self.response(200, {"target": args["target"], "items": items})


# ---------------------------------------------------------------------------
# POST /api/immich/albums/<album_id>/commit
# ---------------------------------------------------------------------------


class ImmichAlbumCommitResource(ProtectedResource, GrampsJSONEncoder):
    """Flow B commit: download, dedup, attach, and tag faces."""

    @api_blueprint.response(201, ImmichAlbumCommitResultSchema())
    @api_blueprint.arguments(ImmichAlbumCommitArgs, location="json")
    def post(self, args: Dict, album_id: str) -> Response:
        """Import the reviewed items into the tree. One DbTxn, all-or-nothing."""
        require_permissions([PERM_EDIT_OBJ])
        db_handle = get_db_handle(readonly=False)

        target = resolve_target(db_handle, args["target"])
        if target is None:
            abort_with_message(404, f"Target object {args['target']!r} not found")
        class_name, target_handle = target

        tree = get_tree_from_jwt()
        media_handler = get_media_handler(db_handle, tree=tree)
        client = ImmichClient()

        try:
            assets_by_id = {a["id"]: a for a in client.album_assets(album_id)}
        except _IMMICH_UNAVAILABLE_ERRORS as exc:
            abort_with_message(502, f"Immich API unavailable: {exc}")

        # ------------------------------------------------------------
        # Pre-validate/download EVERY item before opening the DbTxn, per
        # design doc: corrupt/short-read files are skipped *before* the
        # transaction opens - they must never leave a half-committed
        # transaction. Only after this loop do we touch the database.
        # ------------------------------------------------------------
        prepared: list[Dict[str, Any]] = []
        skipped: list[Dict[str, str]] = []
        for item in args["items"]:
            asset_id = item["immichAssetId"]
            existing_handle = find_media_by_immich_asset_id(db_handle, asset_id)

            if not item.get("attach", True):
                if existing_handle is None:
                    skipped.append(
                        {
                            "immichAssetId": asset_id,
                            "reason": (
                                "attach=false but this asset was never "
                                "previously imported - nothing to attach "
                                "regions to"
                            ),
                        }
                    )
                    continue
                prepared.append(
                    {
                        "asset_id": asset_id,
                        "data": None,
                        "checksum": None,
                        "mime": None,
                        "existing_handle": existing_handle,
                        "regions": item.get("regions", []),
                    }
                )
                continue

            result = fetch_and_validate_asset(client, asset_id)
            if result is None:
                skipped.append(
                    {
                        "immichAssetId": asset_id,
                        "reason": "Download failed or file is corrupt/empty",
                    }
                )
                continue
            data, checksum, _size = result
            mime = _guess_mime(assets_by_id.get(asset_id, {}))
            prepared.append(
                {
                    "asset_id": asset_id,
                    "data": data,
                    "checksum": checksum,
                    "mime": mime,
                    "existing_handle": existing_handle,
                    "regions": item.get("regions", []),
                }
            )

        created_media: list[str] = []
        updated_media: list[str] = []

        with DbTxn("Immich album import", db_handle) as trans:
            set_album_target_mapping(
                db_handle, trans, class_name, target_handle, album_id
            )

            for entry in prepared:
                if entry["data"] is not None:
                    handle, was_created = find_or_create_media(
                        db_handle,
                        trans,
                        media_handler,
                        entry["data"],
                        entry["checksum"],
                        entry["mime"],
                        entry["asset_id"],
                    )
                    (created_media if was_created else updated_media).append(handle)
                    attach_media_to_target(
                        db_handle, trans, class_name, target_handle, handle
                    )
                else:
                    handle = entry["existing_handle"]
                    if handle not in created_media and handle not in updated_media:
                        updated_media.append(handle)

                for region in entry["regions"]:
                    add_person_region(
                        db_handle,
                        trans,
                        region["grampsHandle"],
                        handle,
                        region["rect"],
                    )

        if created_media:
            update_usage_media()

        return self.response(
            201,
            {
                "createdMedia": created_media,
                "updatedMedia": updated_media,
                "skipped": skipped,
            },
        )


# ---------------------------------------------------------------------------
# GET /api/immich/existing/preview
# ---------------------------------------------------------------------------


class ImmichExistingPreviewResource(ProtectedResource, GrampsJSONEncoder):
    """Flow A preview: face suggestions for media already in the tree."""

    @api_blueprint.response(200, ImmichExistingPreviewSchema())
    def get(self) -> Response:
        """Match external-library assets to Gramps Media and propose regions."""
        require_permissions([PERM_EDIT_OBJ])
        db_handle = get_db_handle()
        client = ImmichClient()

        try:
            assets = list(client.list_all_assets())
        except _IMMICH_UNAVAILABLE_ERRORS as exc:
            abort_with_message(502, f"Immich API unavailable: {exc}")

        matches = match_external_assets_to_media(db_handle, assets)

        items = []
        for media_handle, asset in matches:
            asset_id = asset["id"]
            try:
                faces = client.asset_faces(asset_id)
            except _IMMICH_UNAVAILABLE_ERRORS:
                faces = []

            people = []
            for face in faces:
                match = _face_to_person_match(face, db_handle)
                if match is None:
                    continue
                already_has_region = False
                gramps_handle = match["grampsHandle"]
                if gramps_handle:
                    person_obj = db_handle.get_person_from_handle(gramps_handle)
                    already_has_region = person_has_region_on_media(
                        person_obj, media_handle
                    )
                people.append(
                    {
                        "immichPersonId": match["immichPersonId"],
                        "name": match["name"],
                        "grampsHandle": gramps_handle,
                        "proposedRect": match["proposedRect"],
                        "alreadyHasRegion": already_has_region,
                    }
                )

            items.append(
                {
                    "mediaHandle": media_handle,
                    "immichAssetId": asset_id,
                    "people": people,
                }
            )

        return self.response(200, {"items": items})


# ---------------------------------------------------------------------------
# POST /api/immich/existing/commit
# ---------------------------------------------------------------------------


class ImmichExistingCommitResource(ProtectedResource, GrampsJSONEncoder):
    """Flow A commit: add regions to already-in-tree media. Zero-copy."""

    @api_blueprint.response(201, ImmichExistingCommitResultSchema())
    @api_blueprint.arguments(ImmichExistingCommitArgs, location="json")
    def post(self, args: Dict) -> Response:
        """Add reviewed regions to existing media. Files are never touched."""
        require_permissions([PERM_EDIT_OBJ])
        db_handle = get_db_handle(readonly=False)

        updated_media: list[str] = []
        with DbTxn("Immich existing-media region import", db_handle) as trans:
            for item in args["items"]:
                media_handle = item["mediaHandle"]
                any_added = False
                for region in item["regions"]:
                    added = add_person_region(
                        db_handle,
                        trans,
                        region["grampsHandle"],
                        media_handle,
                        region["rect"],
                    )
                    any_added = any_added or added
                if any_added and media_handle not in updated_media:
                    updated_media.append(media_handle)

        return self.response(201, {"updatedMedia": updated_media})


# ---------------------------------------------------------------------------
# GET /api/immich/people/
# ---------------------------------------------------------------------------


class ImmichPeopleResource(ProtectedResource, GrampsJSONEncoder):
    """List Immich people, with their Gramps mapping if any."""

    @api_blueprint.response(200, ImmichPersonSchema(many=True))
    @api_blueprint.arguments(ImmichPeopleQueryArgs, location="query")
    def get(self, args: Dict) -> Response:
        """List/search Immich people for the mapping UI."""
        require_permissions([PERM_EDIT_OBJ])
        db_handle = get_db_handle()
        client = ImmichClient()

        try:
            people = client.list_people()
        except _IMMICH_UNAVAILABLE_ERRORS as exc:
            abort_with_message(502, f"Immich API unavailable: {exc}")

        query = (args.get("query") or "").casefold()
        result = []
        for person in people:
            name = person.get("name", "")
            if query and query not in name.casefold():
                continue
            immich_person_id = person["id"]
            result.append(
                {
                    "immichPersonId": immich_person_id,
                    "name": name,
                    "grampsHandle": find_person_by_immich_id(
                        db_handle, immich_person_id
                    ),
                }
            )
        return self.response(200, result, total_items=len(result))


# ---------------------------------------------------------------------------
# POST /api/immich/mapping/
# ---------------------------------------------------------------------------


class ImmichMappingResource(ProtectedResource, GrampsJSONEncoder):
    """Upsert an Immich person -> Gramps person mapping."""

    @api_blueprint.response(204)
    @api_blueprint.arguments(ImmichMappingArgs, location="json")
    def post(self, args: Dict) -> Response:
        """Persist ``ImmichPersonId`` on the given Gramps person. Never
        auto-linked - always an explicit curator confirmation (design doc
        "Open risks": "mitigated by curator confirmation - never auto-links
        silently").
        """
        require_permissions([PERM_EDIT_OBJ])
        db_handle = get_db_handle(readonly=False)

        with DbTxn("Immich person mapping", db_handle) as trans:
            try:
                set_immich_person_mapping(
                    db_handle, trans, args["grampsHandle"], args["immichPersonId"]
                )
            except ValueError as exc:
                abort_with_message(404, str(exc))

        return Response(status=204)
