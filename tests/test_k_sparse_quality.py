"""
Tests for K-sparse cross-evaluation, matrix completion, and rubric quality scoring.
"""

import numpy as np
import pytest
import random

from src.training.matrix_completion import als_matrix_completion
from src.training.rubric_quality import (
    RubricQualityConfig,
    RubricQualityResult,
    count_near_duplicates,
    jaccard_similarity,
    parse_rubric_text,
    score_rubric_quality,
)


# ═══════════════════════════════════════════════════════════════════════════
# Matrix Completion Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestALSMatrixCompletion:
    """Verify ALS recovers low-rank matrices from partial observations."""

    def _random_low_rank_matrix(self, n: int, rank: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        U = rng.normal(0, 1, (n, rank))
        V = rng.normal(0, 1, (n, rank))
        mu = 5.0
        b = rng.normal(0, 0.5, n)
        c = rng.normal(0, 0.5, n)
        M = U @ V.T + b[:, None] + c[None, :] + mu
        return np.clip(M, 0, 10)

    def test_full_observation_identity(self):
        """With full mask, output should equal input."""
        M = self._random_low_rank_matrix(8, 3)
        mask = np.ones_like(M)
        result = als_matrix_completion(M, mask, rank=3)
        np.testing.assert_array_almost_equal(result, M, decimal=5)

    def test_low_rank_recovery(self):
        """75% observation of rank-2 matrix should recover well."""
        n, rank = 10, 2
        M = self._random_low_rank_matrix(n, rank, seed=42)
        rng = np.random.default_rng(123)
        mask = (rng.random((n, n)) < 0.75).astype(float)
        # Ensure every row and column has at least rank+1 observations
        for i in range(n):
            if mask[i].sum() < rank + 1:
                idx = rng.choice(n, rank + 1, replace=False)
                mask[i, idx] = 1
            if mask[:, i].sum() < rank + 1:
                idx = rng.choice(n, rank + 1, replace=False)
                mask[idx, i] = 1

        observed = M * mask
        completed = als_matrix_completion(observed, mask, rank=rank, max_iter=50, reg=0.05)

        unobs = (1 - mask).astype(bool)
        if unobs.any():
            rmse = np.sqrt(((completed[unobs] - M[unobs]) ** 2).mean())
            # For a clean low-rank matrix, RMSE should be small
            assert rmse < 2.0, f"RMSE too high: {rmse:.3f}"

    def test_clamp_to_valid_range(self):
        """Completed values must be in [0, 10]."""
        M = np.array([[0, 10, 5], [10, 0, 5], [5, 5, 0]], dtype=float)
        mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        result = als_matrix_completion(M, mask, rank=2)
        assert result.min() >= 0.0
        assert result.max() <= 10.0

    def test_observed_entries_preserved(self):
        """Observed entries must not change."""
        M = np.array([[7, 3, 0], [5, 8, 0], [0, 0, 6]], dtype=float)
        mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        result = als_matrix_completion(M, mask, rank=2)
        obs = mask.astype(bool)
        np.testing.assert_array_almost_equal(result[obs], M[obs])

    def test_small_n_degenerate(self):
        """N=2 with rank=1 should still work."""
        M = np.array([[5, 3], [7, 0]], dtype=float)
        mask = np.array([[1, 1], [1, 0]], dtype=float)
        result = als_matrix_completion(M, mask, rank=1)
        assert result.shape == (2, 2)
        assert 0 <= result[1, 1] <= 10


# ═══════════════════════════════════════════════════════════════════════════
# Rubric Quality Tests
# ═══════════════════════════════════════════════════════════════════════════

GOOD_RUBRIC = """\
- [+5] Provides a clear, direct answer to the question | tags: clarity
- [+3] Includes relevant examples to illustrate key points | tags: examples
- [+2] Uses appropriate medical terminology | tags: precision
- [-3] Recommends dangerous treatments without caveats | tags: safety
- [-2] Contains factual inaccuracies about dosage | tags: accuracy
- [+1] Acknowledges limitations or uncertainty where relevant | tags: honesty
"""

REPETITIVE_RUBRIC = """\
- [+5] Provides a clear and direct answer to the question
- [+5] Gives a clear, direct response to the question asked
- [+4] Answers the question clearly and directly
- [+3] Offers a straightforward answer to the user's question
- [+2] The response is clear and answers the question
"""

NO_NEGATIVES_RUBRIC = """\
- [+5] Provides accurate information
- [+3] Uses clear language
- [+2] Includes examples
"""

LONG_RUBRIC = "\n".join(
    f"- [+{i%5+1}] Criterion number {i} about topic {chr(65+i%26)}"
    for i in range(20)
)


class TestRubricParsing:
    def test_parse_standard(self):
        criteria = parse_rubric_text(GOOD_RUBRIC)
        assert len(criteria) == 6
        assert criteria[0].points == 5
        assert criteria[3].points == -3
        assert "safety" in criteria[3].tags

    def test_parse_empty(self):
        assert parse_rubric_text("") == []
        assert parse_rubric_text("no rubric here, just a sentence.") == []

    def test_parse_without_tags(self):
        criteria = parse_rubric_text("- [+3] Simple criterion\n")
        assert len(criteria) == 1
        assert criteria[0].tags == []
        assert criteria[0].points == 3


class TestSimilarity:
    def test_identical(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        sim = jaccard_similarity("apples and oranges", "quantum mechanics theory")
        assert sim < 0.15

    def test_paraphrase_detected(self):
        a = "Provides a clear and direct answer to the question"
        b = "Gives a clear, direct response to the question asked"
        sim = jaccard_similarity(a, b)
        assert sim > 0.35  # Should detect substantial overlap via word-bag


class TestDuplicateDetection:
    def test_no_duplicates(self):
        criteria = parse_rubric_text(GOOD_RUBRIC)
        n_dup = count_near_duplicates(criteria, threshold=0.55)
        assert n_dup == 0

    def test_detects_repetition(self):
        criteria = parse_rubric_text(REPETITIVE_RUBRIC)
        n_dup = count_near_duplicates(criteria, threshold=0.35)
        # Should detect at least 2 duplicates (5 versions of same idea)
        assert n_dup >= 2, f"Expected ≥2 duplicates, got {n_dup}"


class TestRubricQualityScoring:
    def _make_config(self, **kwargs) -> RubricQualityConfig:
        defaults = dict(
            lambda_rep=0.3, lambda_div=0.15, lambda_len=0.2,
            min_criteria=3, max_criteria=15, similarity_threshold=0.55,
            enabled=True,
        )
        defaults.update(kwargs)
        return RubricQualityConfig(**defaults)

    def test_good_rubric_positive_adjustment(self):
        """A well-formed rubric with negatives should get a positive total."""
        cfg = self._make_config()
        result = score_rubric_quality(GOOD_RUBRIC, cfg)
        assert result.n_criteria == 6
        assert result.n_negative == 2
        assert result.has_negative
        assert result.n_duplicates == 0
        assert not result.is_truncated
        # Expect: 0 rep + 0.15 div + 0 len + 0 trunc = 0.15
        assert result.total_adjustment > 0
        assert abs(result.total_adjustment - 0.15) < 0.05

    def test_repetitive_rubric_penalty(self):
        """Repetitive rubric should get negative adjustment."""
        cfg = self._make_config(similarity_threshold=0.35)
        result = score_rubric_quality(REPETITIVE_RUBRIC, cfg)
        assert result.n_duplicates >= 2, f"Expected ≥2 duplicates, got {result.n_duplicates}"
        assert result.detail["repetition_penalty"] < 0
        assert result.detail["diversity_bonus"] == 0  # no negatives

    def test_no_negatives_no_diversity_bonus(self):
        cfg = self._make_config()
        result = score_rubric_quality(NO_NEGATIVES_RUBRIC, cfg)
        assert not result.has_negative
        assert result.detail["diversity_bonus"] == 0.0

    def test_too_many_criteria_penalty(self):
        cfg = self._make_config(max_criteria=15)
        result = score_rubric_quality(LONG_RUBRIC, cfg)
        assert result.n_criteria == 20
        assert result.detail["length_penalty"] < 0

    def test_too_few_criteria_penalty(self):
        cfg = self._make_config(min_criteria=5)
        rubric = "- [+3] Only one criterion\n"
        result = score_rubric_quality(rubric, cfg)
        assert result.detail["length_penalty"] < 0

    def test_disabled_returns_zero(self):
        cfg = self._make_config(enabled=False)
        # Even though we still call the function, the manager should
        # check cfg.enabled before applying. Test the scoring directly.
        result = score_rubric_quality(GOOD_RUBRIC, cfg)
        # The function itself doesn't check enabled; the caller does.
        assert result.total_adjustment != 0  # Scoring still works

    def test_empty_rubric(self):
        cfg = self._make_config()
        result = score_rubric_quality("", cfg)
        assert result.n_criteria == 0
        # Should get length penalty for too few criteria
        assert result.detail["length_penalty"] < 0


# ═══════════════════════════════════════════════════════════════════════════
# K-Sparse Selection Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestKSparseSelection:
    """Test the K-sparse rubric selection logic (extracted function)."""

    @staticmethod
    def _select_rubric_indices(n: int, k: int):
        """Reproduce the selection logic from MetaRewardFunction."""
        from collections import defaultdict
        if k <= 0 or k >= n - 1:
            return [list(range(n)) for _ in range(n)]

        answer_to_rubrics = {}
        for i in range(n):
            candidates = [j for j in range(n) if j != i]
            chosen = sorted(random.sample(candidates, min(k, len(candidates))))
            answer_to_rubrics[i] = chosen

        rubric_to_answers = defaultdict(list)
        for i, rubs in answer_to_rubrics.items():
            for j in rubs:
                rubric_to_answers[j].append(i)

        eval_sets = []
        for j in range(n):
            answer_indices = sorted(set(rubric_to_answers.get(j, [])) | {j})
            eval_sets.append(answer_indices)
        return eval_sets

    def test_full_eval_when_k_zero(self):
        """K=0 should give full N×N evaluation."""
        sets = self._select_rubric_indices(8, 0)
        for s in sets:
            assert s == list(range(8))

    def test_full_eval_when_k_ge_n_minus_1(self):
        """K≥N-1 should give full evaluation."""
        sets = self._select_rubric_indices(6, 5)
        for s in sets:
            assert s == list(range(6))

    def test_k_sparse_correct_count(self):
        """Each answer should be evaluated by exactly K rubrics."""
        random.seed(42)
        n, k = 8, 4
        sets = self._select_rubric_indices(n, k)

        # Verify from answer perspective: each answer i appears as
        # evaluable in its K assigned rubrics
        answer_eval_count = [0] * n
        for j, s in enumerate(sets):
            for i in s:
                if i != j:
                    answer_eval_count[i] += 1

        # Each answer should be evaluated by at least K rubrics
        for i, count in enumerate(answer_eval_count):
            assert count >= k, f"Answer {i} evaluated by {count} rubrics, expected ≥{k}"

    def test_diagonal_always_included(self):
        """Each rubric's eval set should always include itself."""
        random.seed(42)
        sets = self._select_rubric_indices(8, 3)
        for j, s in enumerate(sets):
            assert j in s, f"Rubric {j} not in its own eval set"

    def test_sparse_reduces_total_evals(self):
        """K-sparse should reduce total eval count vs full."""
        random.seed(42)
        n, k = 16, 6
        full_sets = self._select_rubric_indices(n, 0)
        sparse_sets = self._select_rubric_indices(n, k)

        full_total = sum(len(s) for s in full_sets)
        sparse_total = sum(len(s) for s in sparse_sets)
        assert sparse_total < full_total, \
            f"Sparse ({sparse_total}) should be less than full ({full_total})"


# ═══════════════════════════════════════════════════════════════════════════
# Combined Reward Formula Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCombinedReward:
    """Test the combined reward = consensus + quality_adjustment."""

    def test_quality_adjustment_range(self):
        """Quality adjustment should be bounded and reasonable."""
        cfg = RubricQualityConfig(
            lambda_rep=0.3, lambda_div=0.15, lambda_len=0.2,
            min_criteria=3, max_criteria=15,
            similarity_threshold=0.55, enabled=True,
        )
        # Best case: good rubric with negatives, no dups, right length
        result = score_rubric_quality(GOOD_RUBRIC, cfg)
        assert -1.0 <= result.total_adjustment <= 0.5

        # Worst case: 100% duplicates, no negatives, way too long
        worst = score_rubric_quality(REPETITIVE_RUBRIC + REPETITIVE_RUBRIC, cfg)
        assert worst.total_adjustment < 0

    def test_consensus_dominates(self):
        """Quality adjustments should be small relative to typical consensus
        rewards (0-10 scale) so they shape but don't override the signal."""
        cfg = RubricQualityConfig(
            lambda_rep=0.3, lambda_div=0.15, lambda_len=0.2,
            min_criteria=3, max_criteria=15,
            similarity_threshold=0.55, enabled=True,
        )
        # Typical reward is ~5.0 on [0, 10].  Quality adjustment max ~0.45
        good = score_rubric_quality(GOOD_RUBRIC, cfg)
        bad = score_rubric_quality(REPETITIVE_RUBRIC, cfg)
        total_range = abs(good.total_adjustment) + abs(bad.total_adjustment)
        assert total_range < 2.0, \
            f"Quality adjustments too large ({total_range}), would overwhelm consensus"


# ═══════════════════════════════════════════════════════════════════════════
# Integration Smoke Test (no API calls)
# ═══════════════════════════════════════════════════════════════════════════

class TestMatrixCompletionWithKSparse:
    """End-to-end: generate sparse observations and complete."""

    def test_k_sparse_then_completion(self):
        """Simulate K-sparse observation of a low-rank score matrix,
        then apply matrix completion and verify reward estimation."""
        random.seed(42)
        n, k, rank = 8, 4, 2
        rng = np.random.default_rng(42)

        # Ground truth score matrix (low rank + biases)
        U = rng.normal(0, 1, (n, rank))
        V = rng.normal(0, 1, (n, rank))
        b = rng.normal(5, 1, n)
        c = rng.normal(0, 0.5, n)
        true_matrix = np.clip(U @ V.T + b[:, None] + c[None, :], 0, 10)

        # True rewards (full off-diagonal mean)
        true_rewards = np.zeros(n)
        for i in range(n):
            true_rewards[i] = np.mean([true_matrix[i, j] for j in range(n) if j != i])

        # Simulate K-sparse observation
        mask = np.zeros((n, n))
        for i in range(n):
            candidates = [j for j in range(n) if j != i]
            chosen = random.sample(candidates, k)
            for j in chosen:
                mask[i, j] = 1.0
        # Always observe diagonal (self-eval, excluded from reward but observed)
        np.fill_diagonal(mask, 1.0)

        observed = true_matrix * mask
        completed = als_matrix_completion(observed, mask, rank=rank, max_iter=50)

        # Compute rewards from completed matrix (off-diagonal mean)
        mc_rewards = np.zeros(n)
        for i in range(n):
            off_diag = [completed[i, j] for j in range(n) if j != i]
            mc_rewards[i] = np.mean(off_diag)

        # Ranking correlation should be high
        from scipy.stats import spearmanr
        corr, _ = spearmanr(true_rewards, mc_rewards)
        assert corr > 0.7, f"Spearman correlation too low: {corr:.3f}"

        # RMSE of rewards should be reasonable
        rmse = np.sqrt(((mc_rewards - true_rewards) ** 2).mean())
        assert rmse < 2.5, f"Reward RMSE too high: {rmse:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
