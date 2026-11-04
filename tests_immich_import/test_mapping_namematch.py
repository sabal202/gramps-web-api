"""Pure unit tests for immich_import/mapping.py's name-match helpers.

Only the module-level (gramps/gi-free) part of mapping.py is exercised here:
``score_name_match`` and ``suggest_person_matches``. The db-touching helpers
further down in the same file do lazy (function-body) gramps imports, so
loading the whole module by file path (see _loader.py) still works without
gi - those functions just aren't called by these tests.
"""

from _loader import load_module

mapping = load_module("mapping.py", "immich_mapping_under_test")


# ---------------------------------------------------------------------------
# score_name_match
# ---------------------------------------------------------------------------


def test_score_name_match_identical_given_surname_order():
    score = mapping.score_name_match("Иван Петров", "Иван", "Петров")
    assert score == 1.0


def test_score_name_match_surname_given_order():
    # Immich has no given/surname split; a "surname given" order name should
    # still score highly via the full_surname_given comparison.
    score = mapping.score_name_match("Петров Иван", "Иван", "Петров")
    assert score == 1.0


def test_score_name_match_case_and_accent_insensitive():
    score = mapping.score_name_match("jose garcia", "José", "García")
    assert score == 1.0


def test_score_name_match_unrelated_names_score_low():
    # Different script entirely (Cyrillic vs Latin) -> ~zero character
    # overlap and zero token overlap, unlike two Cyrillic names which can
    # share individual letters/spaces by chance (see the "between extremes"
    # test below for a same-script low-but-nonzero case).
    score = mapping.score_name_match("Иван Петров", "John", "Smith")
    assert score < 0.15


def test_score_name_match_partial_overlap_scores_between_extremes():
    # Missing patronymic: Immich name has only two of three tokens.
    full_score = mapping.score_name_match("Иван Петров", "Иван", "Петров")
    partial_score = mapping.score_name_match(
        "Иван Петров", "Иван Сергеевич", "Петров"
    )
    unrelated_score = mapping.score_name_match("Иван Петров", "Анна", "Смирнова")
    assert unrelated_score < partial_score <= full_score


def test_score_name_match_empty_immich_name_scores_zero():
    assert mapping.score_name_match("", "Иван", "Петров") == 0.0


def test_score_name_match_missing_candidate_fields_do_not_crash():
    # name_given/name_surname may be None for a sparse Gramps record.
    score = mapping.score_name_match("Иван Петров", None, None)
    assert score == 0.0


# ---------------------------------------------------------------------------
# suggest_person_matches
# ---------------------------------------------------------------------------


def test_suggest_person_matches_ranks_best_first():
    candidates = [
        {"handle": "h1", "name_given": "Анна", "name_surname": "Смирнова"},
        {"handle": "h2", "name_given": "Иван", "name_surname": "Петров"},
        {"handle": "h3", "name_given": "Иван", "name_surname": "Петровский"},
    ]
    result = mapping.suggest_person_matches("Иван Петров", candidates, min_score=0.0)
    assert result[0]["handle"] == "h2"
    assert result[0]["score"] == 1.0
    # scores are non-increasing
    scores = [c["score"] for c in result]
    assert scores == sorted(scores, reverse=True)


def test_suggest_person_matches_filters_by_min_score():
    candidates = [
        {"handle": "h1", "name_given": "Анна", "name_surname": "Смирнова"},
        {"handle": "h2", "name_given": "Иван", "name_surname": "Петров"},
    ]
    result = mapping.suggest_person_matches(
        "Иван Петров", candidates, min_score=0.9
    )
    assert [c["handle"] for c in result] == ["h2"]


def test_suggest_person_matches_respects_limit():
    candidates = [
        {"handle": f"h{i}", "name_given": "Иван", "name_surname": "Петров"}
        for i in range(10)
    ]
    result = mapping.suggest_person_matches(
        "Иван Петров", candidates, limit=3, min_score=0.0
    )
    assert len(result) == 3


def test_suggest_person_matches_preserves_extra_keys():
    candidates = [
        {
            "handle": "h1",
            "name_given": "Иван",
            "name_surname": "Петров",
            "gramps_id": "I0001",
        }
    ]
    result = mapping.suggest_person_matches("Иван Петров", candidates, min_score=0.0)
    assert result[0]["gramps_id"] == "I0001"
    assert "score" in result[0]


def test_suggest_person_matches_empty_candidates():
    assert mapping.suggest_person_matches("Иван Петров", [], min_score=0.0) == []
