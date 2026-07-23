"""
StrideIQ

scoring_engine.py

Converts raw ML predictions into rich,
interpretable running scores.

This module NEVER performs machine learning.

It enriches ML predictions with:

• confidence
• consistency
• percentile
• severity
• star rating
• coaching priority
• overall score
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List
import math

from cv.models import MetricName
# ==========================================================
# Severity
# ==========================================================

class Severity(Enum):

    ELITE = "Elite"

    EXCELLENT = "Excellent"

    STRONG = "Strong"

    DEVELOPING = "Developing"

    NEEDS_ATTENTION = "Needs Attention"

    HIGH_PRIORITY = "High Priority"

    CRITICAL = "Critical"
# ==========================================================
# Final metric output
# ==========================================================

@dataclass

class MetricResult:

    metric: MetricName

    score: float

    confidence: float

    consistency: float

    percentile: float

    severity: Severity

    stars: float

    improvement_potential: float

    coaching_priority: int
# ==========================================================
# Scoring Engine
# ==========================================================

class ScoringEngine:

    """
    Converts raw predictions

    →

    meaningful coaching metrics.
    """

    def evaluate(

        self,

        metric: MetricName,

        score: float,

        confidence: float,

        consistency: float,

    ) -> MetricResult:

        score = self._clamp(score)

        confidence = self._clamp(confidence * 100) / 100

        consistency = self._clamp(consistency)

        percentile = self._score_to_percentile(score)

        severity = self._severity(score)

        stars = round(score / 20, 1)

        improvement = 100 - score

        priority = self._priority(score)

        return MetricResult(

            metric=metric,

            score=score,

            confidence=confidence,

            consistency=consistency,

            percentile=percentile,

            severity=severity,

            stars=stars,

            improvement_potential=improvement,

            coaching_priority=priority,

        )
# ==========================================================
# Overall score
# ==========================================================

    def overall_score(

        self,

        metrics: List[MetricResult]

    ) -> float:

        if len(metrics) == 0:

            return 0

        weighted_sum = 0

        total_weight = 0

        for m in metrics:

            weight = 0.5 + m.confidence

            weighted_sum += m.score * weight

            total_weight += weight

        return round(

            weighted_sum / total_weight,

            1

        )
# ==========================================================
# Helpers
# ==========================================================

    def _severity(

        self,

        score: float

    ) -> Severity:

        if score >= 95:

            return Severity.ELITE

        if score >= 88:

            return Severity.EXCELLENT

        if score >= 80:

            return Severity.STRONG

        if score >= 70:

            return Severity.DEVELOPING

        if score >= 55:

            return Severity.NEEDS_ATTENTION

        if score >= 40:

            return Severity.HIGH_PRIORITY

        return Severity.CRITICAL

    def _priority(

        self,

        score: float

    ) -> int:

        return int(

            math.ceil(

                (100 - score) / 20

            )

        )

    def _score_to_percentile(

        self,

        score: float

    ) -> float:

        return round(score, 1)

    def _clamp(

        self,

        value: float

    ) -> float:

        return max(

            0,

            min(

                100,

                value

            )

        )