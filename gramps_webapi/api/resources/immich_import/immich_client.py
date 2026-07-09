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

"""Thin Immich REST API client (admin API key auth).

No gramps/gi imports - this module only needs ``requests`` (already a
project dependency, see ``pyproject.toml``) and the stdlib, so it is
importable and instantiable outside the gi-container too (mocked in tests,
per the design doc's testing section: "immich_client -> mocked").

Verified against the v3.0.1 OpenAPI spec (see design doc "Immich API
surface" for the two non-obvious breaking changes from pre-3.0 code
samples this client deliberately works around):

* Album assets are **not** on ``GET /albums/{id}`` (no ``assets`` field in
  v3.0.1, only ``assetCount``) - fetched via ``POST /search/metadata``
  instead.
* Face bounding boxes are **not** on ``AssetResponseDto.people`` (removed in
  v3.0.0, PR #27779) - fetched via ``GET /faces?id=<assetId>`` instead, one
  call per asset.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional

import requests

DEFAULT_BASE_URL = "http://immich_server:2283"
DEFAULT_TIMEOUT = 30
SEARCH_METADATA_PAGE_SIZE = 100


class ImmichApiError(RuntimeError):
    """Raised when the Immich API returns an error response."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Immich API error {status_code}: {message}")


class ImmichClient:
    """Minimal REST client for the subset of the Immich API this cluster uses.

    Reads config from environment variables by default (no Flask app
    context required):

    * ``IMMICH_API_KEY`` - admin-scoped API key (required; raises
      ``ValueError`` if missing and not passed explicitly). Never sent to
      the browser - server-side secret only (see design doc "Open risks").
    * ``IMMICH_API_URL`` - base URL, defaults to
      ``http://immich_server:2283`` (the in-cluster docker service name).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key or os.environ.get("IMMICH_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Immich API key not configured (set IMMICH_API_KEY env var)"
            )
        self.base_url = (base_url or os.environ.get("IMMICH_API_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.timeout = timeout
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "Accept": "application/json"}

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = {**self._headers(), **kwargs.pop("headers", {})}
        resp = self.session.request(
            method, url, headers=headers, timeout=self.timeout, **kwargs
        )
        if resp.status_code >= 400:
            raise ImmichApiError(resp.status_code, resp.text[:500])
        return resp

    # ------------------------------------------------------------------
    # Albums
    # ------------------------------------------------------------------

    def list_albums(self) -> list[dict[str, Any]]:
        """``GET /albums`` - list the archive account's albums.

        Each item has at least ``id``, ``albumName``, ``assetCount``.
        """
        resp = self._request("GET", "/api/albums")
        return resp.json()

    def _search_metadata(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Paginate ``POST /search/metadata`` with an arbitrary filter body,
        following ``nextPage`` until exhausted. Shared by ``album_assets``
        and ``list_all_assets``.
        """
        page: Optional[int] = 1
        while page is not None:
            request_body = {**body, "page": page, "size": SEARCH_METADATA_PAGE_SIZE}
            resp = self._request("POST", "/api/search/metadata", json=request_body)
            data = resp.json()
            assets = data.get("assets", {})
            for item in assets.get("items", []):
                yield item
            page = assets.get("nextPage")

    def album_assets(self, album_id: str) -> Iterator[dict[str, Any]]:
        """Yield every asset in an album via ``POST /search/metadata``.

        ⚠️ Deliberately NOT ``GET /albums/{id}`` - in v3.0.1
        ``AlbumResponseDto`` carries no ``assets`` field, only
        ``assetCount``. Paginates through ``/search/metadata`` with
        ``{"albumIds": [album_id]}`` instead, following ``nextPage`` until
        exhausted.
        """
        yield from self._search_metadata({"albumIds": [album_id]})

    def list_all_assets(self) -> Iterator[dict[str, Any]]:
        """Yield every asset owned by the (archive) account via
        ``POST /search/metadata`` with no album filter.

        Not in the design doc's original 3-method albums/faces/people list,
        but needed for Flow A ("existing tree photos"): matching the
        external-library assets against Gramps Media requires enumerating
        *all* assets, not just one album's. Uses the same paginated search
        endpoint as ``album_assets`` rather than introducing a second API
        surface.
        """
        yield from self._search_metadata({})

    # ------------------------------------------------------------------
    # Faces
    # ------------------------------------------------------------------

    def asset_faces(self, asset_id: str) -> list[dict[str, Any]]:
        """``GET /faces?id=<assetId>`` - bounding boxes for one asset.

        ⚠️ The ``id`` query param is documented as "Face ID" in Immich's
        OpenAPI spec but is actually the **asset** UUID - this is a
        confirmed doc bug, not a typo here. Do not "fix" this to pass a real
        face id; it must stay the asset id or every call 404s.

        Mandatory one-call-per-asset: face boxes were removed from
        ``AssetResponseDto`` in v3.0.0 (PR #27779), so there is no way to
        batch this into the album/search call.

        Returns a list of ``AssetFaceResponseDto``-shaped dicts:
        ``boundingBoxX1/Y1/X2/Y2`` (pixel ints), ``imageWidth``,
        ``imageHeight``, nullable ``person`` (``{id, name, ...}`` or
        ``None`` for an unrecognized/unassigned face).
        """
        resp = self._request("GET", "/api/faces", params={"id": asset_id})
        return resp.json()

    # ------------------------------------------------------------------
    # People
    # ------------------------------------------------------------------

    def list_people(self) -> list[dict[str, Any]]:
        """``GET /people`` - list recognized persons (id/name/thumbnail)."""
        resp = self._request("GET", "/api/people")
        data = resp.json()
        # PeopleResponseDto wraps the list in a "people" key.
        if isinstance(data, dict) and "people" in data:
            return data["people"]
        return data

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_original(self, asset_id: str) -> bytes:
        """``GET /assets/{id}/original`` - raw bytes of the original file.

        Used only for Flow B (new upload -> tree copy). Flow A never
        downloads - it is zero-copy against the external library mount.
        """
        resp = self._request(
            "GET", f"/api/assets/{asset_id}/original", stream=False
        )
        return resp.content

    def get_thumbnail(
        self, asset_id: str, size: str = "preview"
    ) -> tuple[bytes, str]:
        """``GET /assets/{id}/thumbnail`` - thumbnail bytes for one asset.

        Returned as ``(bytes, content_type)``. ``size`` is ``"preview"`` (the
        larger, review-friendly render) or ``"thumbnail"`` (small square).

        Proxied to the browser through our own ``/api/immich/assets/<id>/
        thumbnail`` endpoint so the archive API key never leaves the server
        (design doc "Open risks": the key is server-side only). The browser
        cannot reach ``immich_server`` directly (internal docker hostname,
        no auth), so a same-origin proxy is mandatory.
        """
        resp = self._request(
            "GET",
            f"/api/assets/{asset_id}/thumbnail",
            params={"size": size},
            headers={"Accept": "*/*"},
        )
        return resp.content, resp.headers.get("Content-Type", "image/jpeg")
