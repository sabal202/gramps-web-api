"""Pure unit tests for immich_import/regions.py (no gi required).

Loaded by file path (see _loader.py) so importing this test module never
needs the gramps_webapi package tree, which pulls in gi transitively.
"""

import pytest

from _loader import load_module

regions = load_module("regions.py", "immich_regions_under_test")


# ---------------------------------------------------------------------------
# to_percent_rect
# ---------------------------------------------------------------------------


def test_to_percent_rect_basic():
    # 1000x2000 image, bbox occupies exactly the middle-left quarter-ish.
    rect = regions.to_percent_rect(100, 200, 300, 600, 1000, 2000)
    assert rect == (10, 10, 30, 30)


def test_to_percent_rect_full_image():
    rect = regions.to_percent_rect(0, 0, 1000, 2000, 1000, 2000)
    assert rect == (0, 0, 100, 100)


def test_to_percent_rect_rounds_to_int():
    # 3/7*100 = 42.857... -> rounds to 43
    rect = regions.to_percent_rect(0, 0, 3, 3, 7, 7)
    assert rect == (0, 0, 43, 43)


def test_to_percent_rect_clamps_overflow_beyond_image():
    # bbox extends past the image on all sides.
    rect = regions.to_percent_rect(-500, -500, 1500, 2500, 1000, 2000)
    assert rect == (0, 0, 100, 100)


def test_to_percent_rect_clamps_negative_only():
    rect = regions.to_percent_rect(-100, -50, 200, 400, 1000, 2000)
    left, top, right, bottom = rect
    assert left == 0
    assert top == 0
    assert right == 20
    assert bottom == 20


def test_to_percent_rect_normalizes_inverted_bounds():
    # Malformed/reversed input (x1 > x2): the function must still return an
    # ordered rect (left <= right, top <= bottom) rather than propagate the
    # inversion.
    rect = regions.to_percent_rect(300, 600, 100, 200, 1000, 2000)
    left, top, right, bottom = rect
    assert left <= right
    assert top <= bottom
    assert (left, top, right, bottom) == (10, 10, 30, 30)


@pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_to_percent_rect_raises_on_nonpositive_dims(width, height):
    with pytest.raises(ValueError):
        regions.to_percent_rect(0, 0, 10, 10, width, height)


# ---------------------------------------------------------------------------
# expand_head_shoulders
# ---------------------------------------------------------------------------


def test_expand_head_shoulders_widens_width_by_factor():
    x1, y1, x2, y2 = 100, 100, 200, 300  # width=100, height=200
    nx1, ny1, nx2, ny2 = regions.expand_head_shoulders(
        x1, y1, x2, y2, width_factor=2.0, up_factor=0.0, down_factor=0.0
    )
    assert nx2 - nx1 == pytest.approx(200.0)  # width doubled
    # centered on the same horizontal midpoint
    assert (nx1 + nx2) / 2 == pytest.approx((x1 + x2) / 2)


def test_expand_head_shoulders_up_smaller_than_down_by_default():
    x1, y1, x2, y2 = 100, 100, 200, 300
    nx1, ny1, nx2, ny2 = regions.expand_head_shoulders(x1, y1, x2, y2)
    up_expansion = y1 - ny1
    down_expansion = ny2 - y2
    assert up_expansion > 0
    assert down_expansion > 0
    assert up_expansion < down_expansion, (
        "design requires 'modest up, more down' (forehead vs shoulders)"
    )


def test_expand_head_shoulders_zero_factors_is_noop():
    x1, y1, x2, y2 = 10, 20, 30, 40
    result = regions.expand_head_shoulders(
        x1, y1, x2, y2, width_factor=1.0, up_factor=0.0, down_factor=0.0
    )
    assert result == pytest.approx((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# face_bbox_to_gramps_rect (tight box, no expansion)
# ---------------------------------------------------------------------------


def test_face_bbox_to_gramps_rect_matches_to_percent_rect_directly():
    args = (100, 200, 300, 600, 1000, 2000)
    assert regions.face_bbox_to_gramps_rect(*args) == regions.to_percent_rect(*args)


# ---------------------------------------------------------------------------
# portrait_region_from_face (the function actually used by the endpoints)
# ---------------------------------------------------------------------------


def test_portrait_region_from_face_is_bigger_than_tight_bbox():
    args = (400, 400, 500, 600, 2000, 2000)  # tight 100x200 face, big image
    tight = regions.face_bbox_to_gramps_rect(*args)
    portrait = regions.portrait_region_from_face(*args)

    tight_width = tight[2] - tight[0]
    portrait_width = portrait[2] - portrait[0]
    tight_height = tight[3] - tight[1]
    portrait_height = portrait[3] - portrait[1]

    assert portrait_width > tight_width
    assert portrait_height > tight_height
    # portrait region must still start further up and end further down
    assert portrait[1] <= tight[1]
    assert portrait[3] >= tight[3]


def test_portrait_region_from_face_clamps_near_top_left_corner():
    # A face detected right at the top-left corner: expansion would push
    # past the image edges on the top and left - must clamp, not go negative.
    rect = regions.portrait_region_from_face(0, 0, 50, 80, 1000, 1000)
    left, top, right, bottom = rect
    assert left >= 0
    assert top >= 0
    assert right <= 100
    assert bottom <= 100


def test_portrait_region_from_face_clamps_near_bottom_right_corner():
    rect = regions.portrait_region_from_face(950, 920, 1000, 1000, 1000, 1000)
    left, top, right, bottom = rect
    assert left >= 0
    assert top >= 0
    assert right <= 100
    assert bottom <= 100


def test_portrait_region_from_face_respects_custom_factors():
    args = (400, 400, 500, 600, 2000, 2000)
    default_rect = regions.portrait_region_from_face(*args)
    wider_rect = regions.portrait_region_from_face(*args, width_factor=4.0)
    assert (wider_rect[2] - wider_rect[0]) > (default_rect[2] - default_rect[0])
