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

"""Downstream cluster: import faces/photos from Immich into Gramps.

See ``docs/superpowers/specs/2026-07-09-immich-gramps-import-design.md`` in
the ``gramps-dev`` workspace for the full design. Modules:

``immich_client``
    Thin REST client for the Immich API (admin API key auth).
``regions``
    Pure pixel-bbox -> Gramps-rect-percent math (no gi; unit-tested).
``mapping``
    Immich person/album <-> Gramps handle mapping, stored as Gramps
    Attributes. The name-match helper is pure/unit-tested; the db-touching
    helpers import gramps lazily inside their function bodies.
``gramps_writer``
    Media creation / MediaRef attachment, one ``DbTxn`` per album.
``resources``
    Flask-smorest REST endpoints.
"""
