"""
Rubric quality scoring for RL reward shaping.

Stripped-down version: only two penalties remain:
  1. Criteria count penalty (too many or too few criteria)
  2. Token-level length penalty (quadratic ramp for long rubrics)

All other penalties (repetition, diversity, filler, think leak, garbage chars,
tag repetition, extreme points, non-ASCII, point diversity, truncation) have
been removed to simplify the reward signal.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RubricQualityConfig:
    """Hyperparameters for rubric quality scoring.  All from env vars."""

    # Length penalty: reward -= lambda_len * max(0, n - max_crit) / n
    lambda_len: float = float(os.getenv("GRM_LAMBDA_LEN", "0.2"))
    min_criteria: int = int(os.getenv("GRM_MIN_CRITERIA", "3"))
    max_criteria: int = int(os.getenv("GRM_MAX_CRITERIA", "15"))

    # Token-level length penalty: penalizes rubrics exceeding token_soft_max tokens.
    # penalty = -lambda_token_len * ((tokens - soft_max) / (hard_max - soft_max))^2
    # Quadratic ramp: gentle near soft_max, harsh as tokens approach hard_max.
    lambda_token_len: float = float(os.getenv("GRM_LAMBDA_TOKEN_LEN", "1.5"))
    token_soft_max: int = int(os.getenv("GRM_TOKEN_SOFT_MAX", "750"))
    token_hard_max: int = int(os.getenv("GRM_TOKEN_HARD_MAX", "1024"))

    # Whether to enable quality scoring at all (master switch)
    enabled: bool = os.getenv("GRM_RUBRIC_QUALITY", "true").lower() in ("1", "true", "yes")


# ── Rubric Parsing ─────────────────────────────────────────────────────────

# Matches lines like  "- [+] criterion text | tags: ..."
#                  or  "- [-] criterion text"
#                  or  "- [+3] criterion text" (legacy format, still parseable)
# Tags after | are stripped and ignored.
_CRITERION_RE = re.compile(
    r"^\s*[-*\u2013\u2014]\s*\[([+\-\u2013\u2014](?:\d+)?)\]\s*(.+?)(?:\s*\|\s*tags?\s*:.*)?$",
    re.IGNORECASE,
)


@dataclass
class ParsedCriterion:
    sign: str  # "+" or "-"
    text: str


def parse_rubric_text(rubric_text: str) -> List[ParsedCriterion]:
    """Parse a rubric string into structured criteria.

    Tolerates minor formatting variations (bullet style, spacing).
    Returns an empty list if nothing can be parsed.
    """
    criteria: List[ParsedCriterion] = []
    for line in rubric_text.splitlines():
        m = _CRITERION_RE.match(line)
        if m:
            sign_str = m.group(1)
            sign = "+" if sign_str.startswith("+") else "-"
            text = m.group(2).strip()
            # Normalize en-dash/em-dash bullets to regular hyphen in output
            criteria.append(ParsedCriterion(sign=sign, text=text))
    return criteria


# ── Quality Scoring ────────────────────────────────────────────────────────

@dataclass
class RubricQualityResult:
    """Detailed breakdown of rubric quality analysis."""

    n_criteria: int
    n_positive: int
    n_negative: int
    token_count: int  # actual token count of the rubric (0 if not provided)
    total_adjustment: float
    detail: Dict[str, float]  # component → value


def score_rubric_quality(
    rubric_text: str,
    config: Optional[RubricQualityConfig] = None,
    token_count: int = 0,
) -> RubricQualityResult:
    """Compute a reward adjustment for rubric structural quality.

    Stripped-down version: only criteria count penalty + token length penalty.
    Returns a RubricQualityResult whose `total_adjustment` should be
    ADDED to the base reward.  It can be negative (penalty) or zero.
    """
    if config is None:
        config = RubricQualityConfig()

    criteria = parse_rubric_text(rubric_text)
    n = len(criteria)

    detail: Dict[str, float] = {}

    # ── 1. Length penalty (too many or too few criteria) ────────────────
    if n > config.max_criteria:
        excess = (n - config.max_criteria) / n
        len_penalty = -config.lambda_len * excess
    elif n < config.min_criteria:
        deficit = (config.min_criteria - n) / max(config.min_criteria, 1)
        len_penalty = -config.lambda_len * deficit
    else:
        len_penalty = 0.0
    detail["length_penalty"] = len_penalty

    # ── 2. Token-level length penalty ──────────────────────────────────
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

    n_pos = sum(1 for c in criteria if c.sign == "+")
    n_neg = sum(1 for c in criteria if c.sign == "-")

    total = len_penalty + token_len_penalty

    return RubricQualityResult(
        n_criteria=n,
        n_positive=n_pos,
        n_negative=n_neg,
        token_count=token_count,
        total_adjustment=total,
        detail=detail,
    )
