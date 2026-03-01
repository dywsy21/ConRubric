"""Unit tests for rubric dedup and similarity helpers in grm.py."""
import pytest
from collections import Counter

from src.models.grm import (
    _char_ngrams,
    _jaccard,
    _CRITERION_LINE_RE,
    RubricGenerator,
)


class TestCharNgrams:
    def test_basic(self):
        ng = _char_ngrams("hello", n=3)
        assert ng == Counter({"hel": 1, "ell": 1, "llo": 1})

    def test_empty(self):
        assert _char_ngrams("", n=3) == Counter()

    def test_short(self):
        assert _char_ngrams("ab", n=3) == Counter()

    def test_case_insensitive(self):
        assert _char_ngrams("ABC") == _char_ngrams("abc")


class TestJaccard:
    def test_identical(self):
        a = _char_ngrams("the quick brown fox")
        assert _jaccard(a, a) == 1.0

    def test_disjoint(self):
        a = _char_ngrams("aaa")
        b = _char_ngrams("zzz")
        assert _jaccard(a, b) == 0.0

    def test_empty(self):
        assert _jaccard(Counter(), Counter()) == 0.0

    def test_near_duplicate(self):
        a = _char_ngrams("- [+3] The answer should mention Paris")
        b = _char_ngrams("- [+5] The answer should mention Paris as the capital")
        sim = _jaccard(a, b)
        assert sim > 0.5  # These are clearly similar

    def test_different_criteria(self):
        a = _char_ngrams("- [+3] The answer should mention Paris")
        b = _char_ngrams("- [-5] Must not include outdated information")
        sim = _jaccard(a, b)
        assert sim < 0.4  # These are clearly different


class TestCriterionRegex:
    def test_standard_format(self):
        assert _CRITERION_LINE_RE.match("- [+3] some criterion | tags: accuracy")

    def test_negative_points(self):
        assert _CRITERION_LINE_RE.match("- [-5] bad thing | tags: safety")

    def test_no_brackets(self):
        assert _CRITERION_LINE_RE.match("- +3 some criterion")

    def test_non_criterion(self):
        assert not _CRITERION_LINE_RE.match("This is a header")
        assert not _CRITERION_LINE_RE.match("")


class TestDeduplicateRubric:
    def test_no_duplicates(self):
        rubric = (
            "- [+5] Mentions Paris as capital | tags: accuracy\n"
            "- [-3] Does not hallucinate | tags: safety\n"
            "- [+2] Provides historical context | tags: depth"
        )
        result = RubricGenerator._deduplicate_rubric(rubric)
        assert result.count("\n") == 2  # All 3 lines kept

    def test_removes_near_duplicate(self):
        rubric = (
            "- [+5] The answer should mention Paris as the capital of France | tags: accuracy\n"
            "- [+3] The answer must mention Paris as the capital of France | tags: accuracy\n"
            "- [-3] Does not hallucinate | tags: safety"
        )
        result = RubricGenerator._deduplicate_rubric(rubric, threshold=0.55)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) == 2  # duplicate removed

    def test_keeps_different_criteria(self):
        rubric = (
            "- [+5] Mentions Paris | tags: accuracy\n"
            "- [-3] No hallucination | tags: safety\n"
            "- [+2] Historical context | tags: depth\n"
            "- [-4] Not too verbose | tags: conciseness"
        )
        result = RubricGenerator._deduplicate_rubric(rubric)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) == 4  # All kept

    def test_preserves_non_criterion_lines(self):
        rubric = (
            "## Evaluation Rubric\n"
            "\n"
            "- [+5] Mentions Paris | tags: accuracy\n"
            "- [-3] No hallucination | tags: safety"
        )
        result = RubricGenerator._deduplicate_rubric(rubric)
        assert "## Evaluation Rubric" in result

    def test_empty_input(self):
        assert RubricGenerator._deduplicate_rubric("") == ""

    def test_severe_repetition(self):
        """Simulate the actual pathology: near-identical paraphrases."""
        criteria = [
            "- [+5] The answer should mention Paris as the capital of France",
            "- [+3] The answer must mention Paris as the capital of France",
            "- [+4] The answer should state Paris as the capital of France",
            "- [-5] Must not contain factual errors about geography",
            "- [-3] Should not contain factual errors in geography",
            "- [+6] Answer should be clear and concise",
        ]
        rubric = "\n".join(criteria)
        result = RubricGenerator._deduplicate_rubric(rubric, threshold=0.55)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        # The first 3 are near-identical → keep 1; the 2 negatives are similar → keep 1
        assert len(lines) < 6
        assert len(lines) >= 2  # Keep at least 2 distinct ones
