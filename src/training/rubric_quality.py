"""
Rubric quality scoring for RL reward shaping.

Analyzes generated rubrics for structural quality and penalizes/rewards
properties that correlate with good evaluation rubrics:
  - Repetition penalty: detect near-duplicate criteria via n-gram overlap
  - Diversity bonus: reward rubrics with both positive and negative criteria
  - Length penalty: discourage excessively long/short rubrics
  - Format quality: structural completeness checks

All penalty/bonus coefficients are configurable via environment variables.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RubricQualityConfig:
    """Hyperparameters for rubric quality scoring.  All from env vars."""

    # Repetition penalty: reward -= lambda_rep * (n_duplicates / n_total)
    lambda_rep: float = float(os.getenv("GRM_LAMBDA_REP", "0.3"))

    # Diversity bonus: reward += lambda_div  if rubric has ≥1 negative criterion
    lambda_div: float = float(os.getenv("GRM_LAMBDA_DIV", "0.15"))

    # Length penalty: reward -= lambda_len * max(0, n - max_crit) / n
    lambda_len: float = float(os.getenv("GRM_LAMBDA_LEN", "0.2"))
    min_criteria: int = int(os.getenv("GRM_MIN_CRITERIA", "3"))
    max_criteria: int = int(os.getenv("GRM_MAX_CRITERIA", "15"))

    # Token-level length penalty: penalizes rubrics exceeding token_soft_max tokens.
    # penalty = -lambda_token_len * ((tokens - soft_max) / (hard_max - soft_max))^2
    # Quadratic ramp: gentle near soft_max, harsh as tokens approach hard_max.
    lambda_token_len: float = float(os.getenv("GRM_LAMBDA_TOKEN_LEN", "1.5"))
    token_soft_max: int = int(os.getenv("GRM_TOKEN_SOFT_MAX", "900"))
    token_hard_max: int = int(os.getenv("GRM_TOKEN_HARD_MAX", "1024"))

    # Point diversity bonus: encourages rubrics with varied point values
    # (e.g., +5, +3, -2, -4 rather than all +3).
    # bonus = lambda * min((n_unique_points - 1) / (target - 1), 1.0)
    lambda_point_div: float = float(os.getenv("GRM_LAMBDA_POINT_DIV", "0.3"))
    target_unique_points: int = int(os.getenv("GRM_TARGET_UNIQUE_POINTS", "5"))

    # Similarity threshold for duplicate detection (Jaccard on 3-gram sets)
    similarity_threshold: float = float(os.getenv("GRM_SIM_THRESHOLD", "0.55"))

    # Filler-pattern penalty: penalizes criteria that start with
    # "Models answer with ..." or similar boilerplate.  Per-criterion penalty
    # scaled by (n_filler / n_total).
    lambda_filler: float = float(os.getenv("GRM_LAMBDA_FILLER", "1.5"))

    # Think-tag leakage penalty: penalizes rubrics that contain literal
    # </think> tags in the output text.  These indicate the model is
    # generating internal-monologue artifacts that waste token budget
    # and cause criteria repetition (pre-think criteria get duplicated
    # after the think block).
    # Fixed penalty per occurrence, capped at lambda_think_leak.
    lambda_think_leak: float = float(os.getenv("GRM_LAMBDA_THINK_LEAK", "1.5"))

    # Whether to enable quality scoring at all (master switch)
    enabled: bool = os.getenv("GRM_RUBRIC_QUALITY", "true").lower() in ("1", "true", "yes")


# ── Rubric Parsing ─────────────────────────────────────────────────────────

# Matches lines like  "- [+3] criterion text | tags: ..."
#                  or  "- [-2] criterion text"
_CRITERION_RE = re.compile(
    r"^\s*[-*]\s*\[([+-]?\d+)\]\s*(.+?)(?:\s*\|\s*tags?\s*:\s*(.*))?$",
    re.IGNORECASE,
)


@dataclass
class ParsedCriterion:
    points: int
    text: str
    tags: List[str]


def parse_rubric_text(rubric_text: str) -> List[ParsedCriterion]:
    """Parse a rubric string into structured criteria.

    Tolerates minor formatting variations (bullet style, spacing).
    Returns an empty list if nothing can be parsed.
    """
    criteria: List[ParsedCriterion] = []
    for line in rubric_text.splitlines():
        m = _CRITERION_RE.match(line)
        if m:
            pts = int(m.group(1))
            text = m.group(2).strip()
            tags_str = m.group(3)
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
            criteria.append(ParsedCriterion(points=pts, text=text, tags=tags))
    return criteria


# ── Text Similarity ────────────────────────────────────────────────────────

def _char_ngrams(text: str, n: int = 3) -> set:
    """Return set of character n-grams from lowercased, whitespace-collapsed text."""
    t = re.sub(r"\s+", " ", text.lower().strip())
    if len(t) < n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def _word_ngrams(text: str, n: int = 2) -> set:
    """Return set of word n-grams from lowercased text."""
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _word_bag(text: str) -> set:
    """Return bag of content words (stop-words removed)."""
    _STOP = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "for", "on", "with", "at", "by", "and", "or",
        "that", "this", "it", "its", "from", "as", "not", "but", "if",
        "do", "does", "did", "has", "have", "had", "will", "would", "should",
        "can", "could", "may", "might", "must", "shall",
    })
    words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    return words - _STOP


def jaccard_similarity(a: str, b: str) -> float:
    """Combined similarity using char-3-grams, word-2-grams, and word-bag overlap.

    We take the MAX of three similarity measures to catch both:
    - Near-identical text (char ngrams)
    - Rearranged paraphrases (word bag, word bigrams)
    """
    c_a, c_b = _char_ngrams(a, 3), _char_ngrams(b, 3)
    w_a, w_b = _word_ngrams(a, 2), _word_ngrams(b, 2)
    b_a, b_b = _word_bag(a), _word_bag(b)

    def _jac(s1: set, s2: set) -> float:
        if not s1 and not s2:
            return 1.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0.0

    return max(_jac(c_a, c_b), _jac(w_a, w_b), _jac(b_a, b_b))


def count_near_duplicates(criteria: List[ParsedCriterion],
                          threshold: float = 0.55) -> int:
    """Count criteria that are near-duplicates of an earlier criterion.

    Returns the number of *excess* copies (so 3 copies of the same idea → 2).
    Uses greedy matching: for each criterion, if it is similar to any
    earlier criterion above threshold, it counts as a duplicate.
    """
    n_dup = 0
    for i in range(len(criteria)):
        for j in range(i):
            sim = jaccard_similarity(criteria[i].text, criteria[j].text)
            if sim >= threshold:
                n_dup += 1
                break  # Only count once per duplicate
    return n_dup


# ── Quality Scoring ────────────────────────────────────────────────────────

@dataclass
class RubricQualityResult:
    """Detailed breakdown of rubric quality analysis."""

    n_criteria: int
    n_positive: int
    n_negative: int
    n_duplicates: int
    has_negative: bool
    is_truncated: bool
    token_count: int  # actual token count of the rubric (0 if not provided)
    total_adjustment: float
    detail: Dict[str, float]  # component → value


def score_rubric_quality(
    rubric_text: str,
    config: Optional[RubricQualityConfig] = None,
    token_count: int = 0,
) -> RubricQualityResult:
    """Compute a reward adjustment for rubric structural quality.

    Returns a RubricQualityResult whose `total_adjustment` should be
    ADDED to the cross-consensus reward.  It can be negative (penalty)
    or positive (bonus).
    """
    if config is None:
        config = RubricQualityConfig()

    criteria = parse_rubric_text(rubric_text)
    n = len(criteria)

    detail: Dict[str, float] = {}

    # ── 1. Repetition penalty ──────────────────────────────────────────
    if n >= 2:
        n_dup = count_near_duplicates(criteria, config.similarity_threshold)
        rep_ratio = n_dup / n
        rep_penalty = -config.lambda_rep * rep_ratio
    else:
        n_dup = 0
        rep_penalty = 0.0
    detail["repetition_penalty"] = rep_penalty

    # ── 2. Diversity bonus (has negatives) ─────────────────────────────
    n_pos = sum(1 for c in criteria if c.points > 0)
    n_neg = sum(1 for c in criteria if c.points < 0)
    has_neg = n_neg > 0
    div_bonus = config.lambda_div if has_neg else 0.0
    detail["diversity_bonus"] = div_bonus

    # ── 3. Length penalty (too many or too few criteria) ────────────────
    if n > config.max_criteria:
        excess = (n - config.max_criteria) / n
        len_penalty = -config.lambda_len * excess
    elif n < config.min_criteria:
        deficit = (config.min_criteria - n) / max(config.min_criteria, 1)
        len_penalty = -config.lambda_len * deficit
    else:
        len_penalty = 0.0
    detail["length_penalty"] = len_penalty

    # ── 4. Truncation detection ────────────────────────────────────────
    # Heuristic: if the rubric ends mid-criterion (no closing punctuation,
    # no tag suffix, no complete criterion line), it was likely truncated.
    # A rubric that ends with a complete criterion line (matching the
    # criterion regex) or standard endings is NOT truncated.
    stripped = rubric_text.rstrip()
    if not stripped:
        is_truncated = False
    else:
        last_line = stripped.rsplit("\n", 1)[-1].strip()
        # If last line matches a complete criterion, not truncated
        last_is_criterion = bool(_CRITERION_RE.match(last_line))
        ends_cleanly = stripped.endswith((".", ")", "]", "```"))
        is_truncated = not last_is_criterion and not ends_cleanly
    trunc_penalty = -0.1 if is_truncated else 0.0
    detail["truncation_penalty"] = trunc_penalty

    # ── 5. Token-level length penalty ──────────────────────────────────
    # Quadratic penalty for rubrics exceeding token_soft_max tokens.
    # Ramps from 0 at soft_max to -lambda_token_len at hard_max.
    if token_count > config.token_soft_max:
        excess_ratio = min(
            (token_count - config.token_soft_max)
            / max(config.token_hard_max - config.token_soft_max, 1),
            1.0,
        )
        token_len_penalty = -config.lambda_token_len * (excess_ratio ** 2)
    else:
        token_len_penalty = 0.0
    detail["token_length_penalty"] = token_len_penalty

    # ── 6. Point diversity bonus ───────────────────────────────────
    # Encourage rubrics with varied point values across criteria.
    # More unique point values (e.g., +5, +3, +1, -2, -5) → higher bonus.
    unique_points = set(c.points for c in criteria)
    n_unique = len(unique_points)
    if n_unique >= 2 and config.target_unique_points > 1:
        point_div_bonus = config.lambda_point_div * min(
            (n_unique - 1) / (config.target_unique_points - 1), 1.0
        )
    else:
        point_div_bonus = 0.0
    detail["point_diversity_bonus"] = point_div_bonus

    # ── 7. Filler-pattern penalty ──────────────────────────────────
    # Penalize criteria that follow the "Models answer with ..." template
    # or similar boilerplate filler patterns.  These are a sign the model
    # is generating a generic template instead of question-specific rubric.
    _FILLER_RE = re.compile(
        r"(?i)^models?\s+answers?\s+with\b"
        r"|^(?:the\s+)?(?:response|answer|model)\s+(?:shows?|demonstrates?|maintains?|provides?)\s+"
        r"(?:empathy|compassion|supportive|appropriate|sensitivity)",
    )
    n_filler = sum(1 for c in criteria if _FILLER_RE.search(c.text))
    if n_filler > 0 and n > 0:
        filler_penalty = -config.lambda_filler * (n_filler / n)
    else:
        filler_penalty = 0.0
    detail["filler_pattern_penalty"] = filler_penalty

    # ── 8. Think-tag leakage penalty ───────────────────────────────
    # Detect literal </think> tags in rubric text.  When thinking is
    # disabled, the model sometimes emits these as plain text, followed
    # by filler phrases ("Let me know if you'd like to refine...") and
    # then duplicates the criteria from before the tag.  This wastes
    # token budget and degrades rubric quality.
    #
    # KEY DESIGN: When think-leak is prevalent (most rollouts have it),
    # a flat penalty provides zero discriminative signal.  Instead, we
    # measure the *fraction of the rubric wasted* on think garbage
    # (think tags + filler text + post-think duplicate criteria) so that
    # rubrics with more waste get penalized more, creating gradient even
    # when the base rate is high.
    #
    # Penalty = -lambda * waste_fraction   (0.0 to -lambda)
    think_tag = "<" + "/think>"  # split to avoid matching in source code
    n_think = rubric_text.count(think_tag)

    # Post-think filler patterns (meta-commentary, not criteria)
    _FILLER_PATTERNS = [
        r"(?i)let me know if",
        r"(?i)i think this meets",
        r"(?i)i have completed",
        r"(?i)let me know for revision",
        r"(?i)i think this is good",
        r"(?i)if you\'?d like to refine",
        r"(?i)if you want to adjust",
    ]
    n_filler_phrases = sum(1 for p in _FILLER_PATTERNS if re.search(p, rubric_text))

    if n_think > 0 or n_filler_phrases > 0:
        # Estimate wasted characters: everything after the FIRST think tag
        # is likely duplicated criteria + filler.  Also count filler phrase
        # lengths even if no think tag is present.
        first_think_pos = rubric_text.find(think_tag)
        if first_think_pos >= 0:
            # Everything after first think tag is waste
            waste_chars = len(rubric_text) - first_think_pos
        else:
            # No think tag but has filler phrases — estimate ~50 chars each
            waste_chars = n_filler_phrases * 50

        total_chars = max(len(rubric_text), 1)
        waste_fraction = min(waste_chars / total_chars, 1.0)

        # Scale: penalty grows with waste fraction
        think_penalty = -config.lambda_think_leak * waste_fraction
    else:
        think_penalty = 0.0
    detail["think_leak_penalty"] = think_penalty
    detail["think_leak_count"] = n_think
    detail["think_filler_count"] = n_filler_phrases

    total = (rep_penalty + div_bonus + len_penalty + trunc_penalty
             + token_len_penalty + point_div_bonus + filler_penalty
             + think_penalty)

    return RubricQualityResult(
        n_criteria=n,
        n_positive=n_pos,
        n_negative=n_neg,
        n_duplicates=n_dup,
        has_negative=has_neg,
        is_truncated=is_truncated,
        token_count=token_count,
        total_adjustment=total,
        detail=detail,
    )
