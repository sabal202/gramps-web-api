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

"""Gramps DB writes for the Immich import cluster.

⚠️ gi-dependent - cannot be imported or run on Windows (no ``gi``/PyGObject).
Only ``python -m py_compile`` was used to catch syntax errors locally; the
actual write logic must be verified in the gi-container on the NAS, per
memory ``gramps-native-write-import`` and ``gramps-forks-build-deploy``.

Every function here that mutates the DB takes a caller-provided ``trans``
(``DbTxn``) - callers (``resources.py``) are expected to open exactly ONE
``DbTxn`` per album commit and pass it through every helper call, per the
design doc's "all-or-nothing" requirement. Corrupt/short-read downloads must
be filtered out via ``fetch_and_validate_asset`` *before* the ``DbTxn`` opens
- never call the create/attach helpers with unvalidated data.

Person-region convention (confirmed against Gramps' own gallery model): a
"face tag" is a ``MediaRef`` with a crop ``rect`` added to the **person's
own** ``media_list`` - not a rect stored on the photo pointing at the
person. This matches how Gramps Desktop's own "Add tag" gallery workflow
works and how the frontend's ``renderPersonListItem``/avatar code reads
``person.media_list`` (see ``gramps-web`` conventions in this workspace's
CLAUDE.md, "Union status"/kinship person payloads use exactly this shape:
``{"ref": ..., "rect": ...}``).
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Optional

from .mapping import (
    ATTR_IMMICH_ASSET_ID,
    get_attribute_value,
    set_attribute_value,
)

IMMICH_IMPORT_SUBDIR = "immich-import"


def fetch_and_validate_asset(
    client: Any, asset_id: str
) -> Optional[tuple[bytes, str, int]]:
    """Download and checksum one Immich asset's original file.

    Returns ``(data, checksum, size)`` or ``None`` if the download failed or
    the file is empty/corrupt (Phase-0 testing saw premature-JPEG/short-read
    cases) - callers must call this for every item BEFORE opening the
    ``DbTxn``, and route ``None`` results to the commit response's
    ``skipped`` list rather than aborting the whole album.
    """
    from ...file import process_file  # gramps_webapi.api.file

    try:
        data = client.download_original(asset_id)
    except Exception:  # noqa: BLE001 - any transport/HTTP error -> skip, don't crash the album
        return None
    if not data:
        return None
    try:
        checksum, size, _fp = process_file(BytesIO(data))
    except IOError:
        return None
    return data, checksum, size


def find_media_by_checksum(db_handle: Any, checksum: str) -> Optional[Any]:
    """Return the existing ``Media`` object with this checksum, or ``None``.

    O(n) scan over ``iter_media()`` - there is no checksum index. Acceptable
    for the tree sizes this deploys against; revisit if that stops being
    true (same tradeoff noted in ``mapping.find_person_by_immich_id``).
    """
    for media in db_handle.iter_media():
        if media.checksum == checksum:
            return media
    return None


def create_media_for_import(
    db_handle: Any,
    trans: Any,
    media_handler: Any,
    data: bytes,
    checksum: str,
    mime: str,
    immich_asset_id: str,
) -> str:
    """Copy *data* into ``<media_base>/immich-import/<checksum>.<ext>``,
    create a new ``Media`` object tagged with ``ImmichAssetId``, commit it
    within *trans*, and return the new handle.

    Caller must already have confirmed no existing Media has this checksum
    (see ``find_media_by_checksum`` / ``find_or_create_media``).
    """
    from gramps.gen.lib import Media

    from ..util import add_object  # gramps_webapi.api.resources.util

    default_name = media_handler.get_default_filename(checksum, mime)
    rel_path = f"{IMMICH_IMPORT_SUBDIR}/{default_name}"
    media_handler.upload_file(BytesIO(data), checksum, mime, path=rel_path)

    obj = Media()
    obj.set_checksum(checksum)
    obj.set_path(rel_path)
    obj.set_mime_type(mime)
    set_attribute_value(obj, ATTR_IMMICH_ASSET_ID, immich_asset_id)

    try:
        add_object(db_handle, obj, trans)
    except ValueError as exc:
        raise ValueError(f"Failed to add Media for asset {immich_asset_id}") from exc
    return obj.handle


def find_or_create_media(
    db_handle: Any,
    trans: Any,
    media_handler: Any,
    data: bytes,
    checksum: str,
    mime: str,
    immich_asset_id: str,
) -> tuple[str, bool]:
    """Dedup-aware Media creation. Returns ``(handle, created)``.

    If a Media with this checksum already exists, reuse it (tagging it with
    ``ImmichAssetId`` if it didn't already carry one - e.g. it was imported
    by hand before this cluster existed) instead of copying the file again.
    """
    existing = find_media_by_checksum(db_handle, checksum)
    if existing is not None:
        from ..util import update_object  # gramps_webapi.api.resources.util

        if get_attribute_value(existing, ATTR_IMMICH_ASSET_ID) != immich_asset_id:
            set_attribute_value(existing, ATTR_IMMICH_ASSET_ID, immich_asset_id)
            update_object(db_handle, existing, trans)
        return existing.handle, False

    handle = create_media_for_import(
        db_handle, trans, media_handler, data, checksum, mime, immich_asset_id
    )
    return handle, True


def attach_media_to_target(
    db_handle: Any, trans: Any, class_name: str, target_handle: str, media_handle: str
) -> bool:
    """Attach *media_handle* as a plain (no-rect) ``MediaRef`` on the target
    object (event/family/person/... - whatever ``class_name`` names).

    Returns ``True`` if a new reference was added, ``False`` if the target
    already referenced this media (idempotent re-import of the same album).
    """
    from gramps.gen.lib import MediaRef

    from ..util import update_object  # gramps_webapi.api.resources.util

    get_method = db_handle.method("get_%s_from_handle", class_name)
    if get_method is None:
        raise ValueError(f"Unknown target class {class_name!r}")
    obj = get_method(target_handle)
    if obj is None:
        raise ValueError(f"{class_name} handle {target_handle!r} not found")

    if any(ref.ref == media_handle for ref in obj.get_media_list()):
        return False

    media_ref = MediaRef()
    media_ref.set_reference_handle(media_handle)
    obj.add_media_reference(media_ref)
    update_object(db_handle, obj, trans)
    return True


def person_has_region_on_media(person: Any, media_handle: str) -> bool:
    """True if *person* already has a cropped ``MediaRef`` (a "face tag")
    pointing at *media_handle*.

    Used by the Flow-A preview to only propose regions "where the person has
    no rect on that media yet" (design doc, Flow A description).
    """
    return any(
        ref.ref == media_handle and ref.rect for ref in person.get_media_list()
    )


def add_person_region(
    db_handle: Any,
    trans: Any,
    gramps_person_handle: str,
    media_handle: str,
    rect: tuple[int, int, int, int],
) -> bool:
    """Add a cropped ``MediaRef`` (face tag) for *rect* to the person's own
    ``media_list``. Returns ``True`` if added, ``False`` if an identical
    ``(media_handle, rect)`` reference already existed (idempotent against
    re-submitting the same commit).

    Gramps convention: the crop lives on the PERSON's MediaRef into the
    photo, not on the photo pointing at the person - see module docstring.
    """
    from gramps.gen.errors import HandleError
    from gramps.gen.lib import MediaRef

    from ..util import update_object  # gramps_webapi.api.resources.util

    try:
        person = db_handle.get_person_from_handle(gramps_person_handle)
    except HandleError as exc:
        raise ValueError(
            f"Person handle {gramps_person_handle!r} not found"
        ) from exc
    if person is None:
        raise ValueError(f"Person handle {gramps_person_handle!r} not found")

    rect_tuple = tuple(int(v) for v in rect)
    for ref in person.get_media_list():
        if ref.ref == media_handle and ref.rect == rect_tuple:
            return False

    media_ref = MediaRef()
    media_ref.set_reference_handle(media_handle)
    media_ref.set_rectangle(rect_tuple)
    person.add_media_reference(media_ref)
    update_object(db_handle, person, trans)
    return True
