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

"""Pure region math: Immich pixel face bbox -> Gramps percent rect.

No imports from gramps/gi anywhere in this module - it is plain arithmetic
on floats/ints and is unit-tested standalone (see
``tests_immich_import/test_regions.py``, loaded by file path so the test
never has to import the ``gramps_webapi`` package tree, which pulls in
``gi`` transitively on import).

Gramps' ``MediaRef.rect`` is ``[left, top, right, bottom]`` as **percentages
of the image (0-100), stored as ints** (see ``gramps.gen.lib.mediaref`` JSON
schema: ``rect`` is an array of 4 integers). Immich's ``GET /faces?id=``
gives a **pixel** bounding box (``boundingBoxX1/Y1/X2/Y2``) plus the
``imageWidth``/``imageHeight`` of the record the box was measured against.

⚠️ EXIF-orientation caveat (see design doc "Tricky points" #1): the
``imageWidth``/``imageHeight`` Immich returns describe its ML **preview**
image, not necessarily the original file as Gramps will render it. If the
preview was auto-rotated per EXIF orientation while Gramps displays the
original without applying that rotation (or vice versa), a percent rect
computed here can land rotated/mirrored relative to what the curator sees.
This module has no way to detect that mismatch - it trusts the width/height
it is given. It must be validated against a real phone photo with
orientation != 1 before this math is trusted broadly; the editable review
step (curator eyeballs/adjusts the drawn box before commit) is the backstop.
"""

from __future__ import annotations

from typing import NamedTuple

# "Head and shoulders" expansion defaults, tuned against Immich's tight
# face-detector boxes. Asymmetric on purpose: faces are detected roughly
# eyebrow to chin, so a modest bump above covers forehead/hair, while a much
# larger bump below is needed to reach the shoulders. Width is expanded less
# aggressively than height since faces are usually taller than the desired
# margin needs to be wide.
DEFAULT_WIDTH_FACTOR = 2.2
DEFAULT_UP_FACTOR = 0.5
DEFAULT_DOWN_FACTOR = 1.5

PERCENT_MIN = 0
PERCENT_MAX = 100


class PixelBBox(NamedTuple):
    """A pixel-space bounding box as returned by ``GET /faces?id=``."""

    x1: float
    y1: float
    x2: float
    y2: float


def _validate_dims(image_width: float, image_height: float) -> None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image_width/image_height must be positive, got "
            f"{image_width!r}/{image_height!r}"
        )


def to_percent_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: float,
    image_height: float,
) -> tuple[int, int, int, int]:
    """Convert a pixel bbox to a Gramps rect ``[left, top, right, bottom]``.

    The result is in percent of the image (0-100), rounded to ints and
    clamped to the image edges. Coordinates are normalized so
    ``left <= right`` and ``top <= bottom`` even if the input was not
    (can happen after the head+shoulders expansion pushes a corner past
    an edge for very small/near-edge faces).
    """
    _validate_dims(image_width, image_height)

    left = x1 / image_width * 100.0
    top = y1 / image_height * 100.0
    right = x2 / image_width * 100.0
    bottom = y2 / image_height * 100.0

    left = min(max(left, PERCENT_MIN), PERCENT_MAX)
    top = min(max(top, PERCENT_MIN), PERCENT_MAX)
    right = min(max(right, PERCENT_MIN), PERCENT_MAX)
    bottom = min(max(bottom, PERCENT_MIN), PERCENT_MAX)

    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    return (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )


def expand_head_shoulders(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    width_factor: float = DEFAULT_WIDTH_FACTOR,
    up_factor: float = DEFAULT_UP_FACTOR,
    down_factor: float = DEFAULT_DOWN_FACTOR,
) -> tuple[float, float, float, float]:
    """Expand a tight pixel face bbox into a "head + shoulders" portrait box.

    Still returns **pixel** coordinates (not yet converted to percent, not
    yet clamped to the image) - callers pass the result to
    ``to_percent_rect`` for that. Kept as a separate step so
    ``portrait_region_from_face`` (see below) and any future caller that
    wants to inspect/tweak the pixel-space box before conversion can do so.

    ``width_factor`` multiplies the original face width, centered on the
    face's horizontal midpoint. ``up_factor``/``down_factor`` are fractions
    of the original face *height* added above/below the face box
    respectively (asymmetric: modest up, more down).
    """
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) / 2.0

    new_half_width = (width * width_factor) / 2.0
    new_x1 = cx - new_half_width
    new_x2 = cx + new_half_width

    new_y1 = y1 - height * up_factor
    new_y2 = y2 + height * down_factor

    return new_x1, new_y1, new_x2, new_y2


def face_bbox_to_gramps_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: float,
    image_height: float,
) -> tuple[int, int, int, int]:
    """Convert a tight Immich face bbox directly to a Gramps percent rect.

    No head+shoulders expansion - use this when the tight face box itself
    is wanted (e.g. for debugging/inspection). Flow A/B review use
    ``portrait_region_from_face`` instead.
    """
    return to_percent_rect(x1, y1, x2, y2, image_width, image_height)


def portrait_region_from_face(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: float,
    image_height: float,
    *,
    width_factor: float = DEFAULT_WIDTH_FACTOR,
    up_factor: float = DEFAULT_UP_FACTOR,
    down_factor: float = DEFAULT_DOWN_FACTOR,
) -> tuple[int, int, int, int]:
    """Immich pixel face bbox -> "head + shoulders" Gramps percent rect.

    This is the function the review/commit endpoints use to seed the
    editable rect for each detected face: expand the tight face box
    asymmetrically (more downward, for shoulders), then convert to percent
    and clamp to the image edges.
    """
    ex1, ey1, ex2, ey2 = expand_head_shoulders(
        x1,
        y1,
        x2,
        y2,
        width_factor=width_factor,
        up_factor=up_factor,
        down_factor=down_factor,
    )
    return to_percent_rect(ex1, ey1, ex2, ey2, image_width, image_height)
