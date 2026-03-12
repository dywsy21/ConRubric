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

    # Malformed criterion penalty: penalize lines that look like criteria but have
    # invalid bracket syntax (e.g., [1+], [−1−] instead of [+], [-]).
    # This punishes format drift strongly.
    lambda_malformed: float = float(os.getenv("GRM_LAMBDA_MALFORMED", "2.0"))

    # Whether to enable quality scoring at all (master switch)
    enabled: bool = os.getenv("GRM_RUBRIC_QUALITY", "true").lower() in ("1", "true", "yes")


# ── Rubric Parsing ─────────────────────────────────────────────────────────

# Matches lines like  "- [+] criterion text | tags: ..."
#                  or  "- [-] criterion text"
#                  or  "- [+3] criterion text" (legacy format)
#                  or  "- [1+] criterion text" (digit-first format, model drift)
#                  or  "- [−1−] criterion text" (digit-first with en-dash)
# Tags after | are stripped and ignored.
# Format 1 (sign-first): [+], [-], [+3], [-2], [−1] (en-dash allowed)
# Format 2 (digit-first): [1+], [2-], [1−] (model sometimes generates this)
# IMPORTANT: At least one sign (+/-/en-dash/em-dash) MUST be present.
# The leading bullet (- or *) is optional to support base models that omit it.
_CRITERION_RE = re.compile(
    r"^\s*(?:[-*\u2013\u2014]\s*)?\[(\d*[+\-\u2013\u2014]+\d*)\]\s*(.+?)(?:\s*\|\s*tags?\s*:.*)?$",
    re.IGNORECASE,
)

# Regex to detect lines that LOOK like criteria but have malformed bracket content
# Used to penalize format drift. Matches optional bullet + bracket but invalid content.
_MALFORMED_CRITERION_RE = re.compile(
    r"^\s*(?:[-*\u2013\u2014]\s*)?\[[^\]]*\]\s*.+$",
)


@dataclass
class ParsedCriterion:
    sign: str  # "+" or "-"
    text: str


def _extract_sign(bracket_content: str) -> str:
    """Extract sign from bracket content, handling both sign-first and digit-first formats.
    
    Examples:
        "+" -> "+", "-" -> "-"
        "+3" -> "+", "-2" -> "-"
        "1+" -> "+", "2-" -> "-"
        "−1" -> "-" (en-dash)
        "1−" -> "-" (digit-first with en-dash)
    """
    # Normalize en-dash/em-dash to regular minus
    normalized = bracket_content.replace("\u2013", "-").replace("\u2014", "-")
    
    # Check for explicit sign anywhere
    if "+" in normalized:
        return "+"
    elif "-" in normalized:
        return "-"
    else:
        # No sign found - default to positive (shouldn't happen with valid format)
        return "+"


def parse_rubric_text(rubric_text: str) -> List[ParsedCriterion]:
    """Parse a rubric string into structured criteria.

    Tolerates minor formatting variations (bullet style, spacing, sign position).
    Returns an empty list if nothing can be parsed.
    """
    criteria: List[ParsedCriterion] = []
    for line in rubric_text.splitlines():
        m = _CRITERION_RE.match(line)
        if m:
            bracket_content = m.group(1)
            sign = _extract_sign(bracket_content)
            text = m.group(2).strip()
            criteria.append(ParsedCriterion(sign=sign, text=text))
    return criteria


def count_malformed_criteria(rubric_text: str) -> int:
    """Count lines that look like criteria but have malformed bracket content.
    
    This detects format drift where the model generates invalid bracket syntax
    like [1+], [−1−], etc. These lines LOOK like criteria but won't parse.
    
    Returns count of lines that match bullet+bracket pattern but fail strict parsing.
    """
    malformed = 0
    for line in rubric_text.splitlines():
        # Check if line looks like a criterion (bullet + brackets)
        if _MALFORMED_CRITERION_RE.match(line):
            # But doesn't match the strict criterion pattern
            if not _CRITERION_RE.match(line):
                malformed += 1
    return malformed


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

    # ── 3. Malformed criterion penalty ──────────────────────────────────
    # Strongly penalize lines that look like criteria but have invalid format.
    # This catches format drift like [1+], [−1−] which the model may generate.
    n_malformed = count_malformed_criteria(rubric_text)
    if n_malformed > 0:
        # Penalty proportional to number of malformed lines, capped at -lambda_malformed
        malformed_penalty = -config.lambda_malformed * min(n_malformed / 3.0, 1.0)
    else:
        malformed_penalty = 0.0
    detail["malformed_penalty"] = malformed_penalty

    n_pos = sum(1 for c in criteria if c.sign == "+")
    n_neg = sum(1 for c in criteria if c.sign == "-")

    total = len_penalty + token_len_penalty + malformed_penalty

    return RubricQualityResult(
        n_criteria=n,
        n_positive=n_pos,
        n_negative=n_neg,
        token_count=token_count,
        total_adjustment=total,
        detail=detail,
    )
