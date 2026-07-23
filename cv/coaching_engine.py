"""
StrideIQ

coaching_engine.py

Transforms biomechanical scores into
high-level coaching insights.

This module is the expert running coach
inside StrideIQ.

NO machine learning.
NO LLM.

Only biomechanical reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

from cv.models import MetricName
from cv.scoring import MetricResult

# ============================================================
# Coaching Domains
# ============================================================

class Domain(Enum):
    UPPER_BODY = "Upper Body"
    LOWER_BODY = "Lower Body"
    WHOLE_BODY = "Whole Body"
    FATIGUE = "Fatigue"
    SENSOR_FUSION = "Sensor Fusion"

# ============================================================
# Severity Levels
# ============================================================

class Severity(Enum):
    ELITE = "Elite"
    STRONG = "Strong"
    DEVELOPING = "Developing"
    NEEDS_ATTENTION = "Needs Attention"
    HIGH_PRIORITY = "High Priority"

# ============================================================
# Coaching Finding
# ============================================================

@dataclass
class Finding:
    id: str
    title: str
    domain: Domain
    confidence: float
    severity: Severity
    priority: int
    explanation: str
    recommendation: str
    evidence: List[str]
    wearable_supported: bool
    cv_supported: bool

# ============================================================
# Coaching Engine
# ============================================================

class CoachingEngine:
    """
    Expert biomechanical reasoning engine.

    Input:
        MetricResults
        Wearable metrics

    Output:
        Ranked coaching findings.
    """

    def __init__(self):
        self.metric_map: Dict[MetricName, MetricResult] = {}
        self.findings: List[Finding] = []

    # --------------------------------------------------------

    def generate(
        self,
        metrics: List[MetricResult],
        wearable_data: Optional[Dict] = None,
    ) -> List[Finding]:
        self.metric_map = {
            m.metric: m
            for m in metrics
        }

        self.findings = []

        # Upper Body
        self.upper_body_rules()

        # Lower Body
        self.lower_body_rules()

        # Whole Body
        self.whole_body_rules()

        # Fatigue
        self.fatigue_rules()

        # Wearables
        if wearable_data is not None:
            self.sensor_fusion_rules(wearable_data)

        self.findings.sort(
            key=lambda x: (
                x.priority,
                x.confidence
            ),
            reverse=True
        )

        return self.findings

    # --------------------------------------------------------

    def metric(self, metric: MetricName) -> MetricResult:
        return self.metric_map[metric]

    # --------------------------------------------------------

    def add_finding(
        self,
        *,
        id: str,
        title: str,
        domain: Domain,
        confidence: float,
        severity: Severity,
        evidence: List[str],
        explanation: str,
        recommendation: str,
        priority: int,
        wearable_supported: bool = False,
        cv_supported: bool = True,
    ):
        self.findings.append(
            Finding(
                id=id,
                title=title,
                domain=domain,
                confidence=confidence,
                severity=severity,
                evidence=evidence,
                explanation=explanation,
                recommendation=recommendation,
                priority=priority,
                wearable_supported=wearable_supported,
                cv_supported=cv_supported,
            )
        )

    # =======================================================
    # Rule Groups
    # =======================================================

    def upper_body_rules(self):
        shoulder = self.metric(MetricName.SHOULDER_RELAXATION)
        arm = self.metric(MetricName.ARM_SWING)
        crossover = self.metric(MetricName.ARM_CROSSOVER)

        if arm.score < 65 and crossover.score < 65:
            self.add_finding(
                id="UB001",
                title="Upper Body Compensation",
                domain=Domain.UPPER_BODY,
                confidence=0.94,
                severity=Severity.HIGH_PRIORITY,
                evidence=[
                    "Restricted arm swing",
                    "Excessive arm crossover"
                ],
                explanation=(
                    "The arms are compensating by "
                    "crossing the body's midline "
                    "instead of driving backward."
                ),
                recommendation=(
                    "Relax your shoulders and "
                    "focus on driving your elbows "
                    "behind you."
                ),
                priority=5,
                wearable_supported=False,
                cv_supported=True
            )

        if shoulder.score < 65 and arm.score < 70:
            self.add_finding(
                id="UB002",
                title="Shoulder Tension",
                domain=Domain.UPPER_BODY,
                confidence=0.91,
                severity=Severity.NEEDS_ATTENTION,
                evidence=[
                    "Shoulder relaxation reduced",
                    "Restricted arm swing"
                ],
                explanation=(
                    "Shoulder tension is likely "
                    "limiting natural arm motion."
                ),
                recommendation=(
                    "Keep your shoulders relaxed "
                    "and allow your arms to swing "
                    "freely."
                ),
                priority=4,
                wearable_supported=False,
                cv_supported=True
            )

        if shoulder.score > 90 and arm.score > 85:
            self.add_finding(
                id="UB003",
                title="Excellent Upper Body Mechanics",
                domain=Domain.UPPER_BODY,
                confidence=0.97,
                severity=Severity.ELITE,
                evidence=[
                    "Relaxed shoulders",
                    "Efficient arm swing"
                ],
                explanation=(
                    "Upper-body mechanics are "
                    "efficient and unlikely to "
                    "limit running performance."
                ),
                recommendation=(
                    "Maintain your current "
                    "upper-body technique."
                ),
                priority=1,
                wearable_supported=False,
                cv_supported=True
            )

        if crossover.score < 60 and shoulder.score > 80:
            self.add_finding(
                id="UB004",
                title="Cross-body Arm Swing",
                domain=Domain.UPPER_BODY,
                confidence=0.88,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Arm crossover detected"
                ],
                explanation=(
                    "Shoulders are relaxed but "
                    "the arms are travelling "
                    "across the torso."
                ),
                recommendation=(
                    "Keep your hands moving "
                    "forward and backward rather "
                    "than side to side."
                ),
                priority=3,
                wearable_supported=False,
                cv_supported=True
            )

        if arm.consistency < 60:
            self.add_finding(
                id="UB005",
                title="Inconsistent Arm Rhythm",
                domain=Domain.UPPER_BODY,
                confidence=0.86,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Arm swing consistency dropped"
                ],
                explanation=(
                    "Arm mechanics changed "
                    "throughout the run."
                ),
                recommendation=(
                    "Maintain a consistent arm "
                    "rhythm even during fatigue."
                ),
                priority=3,
                wearable_supported=False,
                cv_supported=True
            )

        if arm.score > 85 and crossover.score > 85:
            self.add_finding(
                id="UB006",
                title="Efficient Arm Mechanics",
                domain=Domain.UPPER_BODY,
                confidence=0.95,
                severity=Severity.STRONG,
                evidence=[
                    "Strong arm drive",
                    "Minimal crossover"
                ],
                explanation=(
                    "Your arm mechanics are efficient, "
                    "helping maintain forward momentum "
                    "with minimal wasted motion."
                ),
                recommendation=(
                    "Maintain this relaxed arm rhythm."
                ),
                priority=1,
            )

        if shoulder.score > 85 and arm.score < 60:
            self.add_finding(
                id="UB007",
                title="Possible Elbow Drive Restriction",
                domain=Domain.UPPER_BODY,
                confidence=0.90,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Relaxed shoulders",
                    "Poor arm swing"
                ],
                explanation=(
                    "Shoulder mobility appears good, "
                    "so the restriction is likely in "
                    "the elbow drive."
                ),
                recommendation=(
                    "Focus on driving the elbows "
                    "backwards rather than moving "
                    "the hands."
                ),
                priority=3,
            )

        if shoulder.consistency > 85 and arm.consistency > 85:
            self.add_finding(
                id="UB008",
                title="Stable Upper Body",
                domain=Domain.UPPER_BODY,
                confidence=0.96,
                severity=Severity.ELITE,
                evidence=[
                    "Consistent shoulders",
                    "Consistent arm rhythm"
                ],
                explanation=(
                    "Upper-body movement remained "
                    "stable throughout the run."
                ),
                recommendation=(
                    "Continue maintaining this "
                    "consistent rhythm."
                ),
                priority=1,
            )

    def lower_body_rules(self):
        knee = self.metric(MetricName.KNEE_DRIVE)

        if knee.score < 60:
            self.add_finding(
                id="LB001",
                title="Weak Knee Drive",
                domain=Domain.LOWER_BODY,
                confidence=0.90,
                severity=Severity.NEEDS_ATTENTION,
                evidence=[
                    "Reduced knee lift"
                ],
                explanation=(
                    "The knees are not driving "
                    "high enough to generate "
                    "strong propulsion."
                ),
                recommendation=(
                    "Focus on lifting the knee "
                    "forward rather than pushing "
                    "down."
                ),
                priority=4,
            )

        if knee.score > 90:
            self.add_finding(
                id="LB002",
                title="Excellent Knee Drive",
                domain=Domain.LOWER_BODY,
                confidence=0.97,
                severity=Severity.ELITE,
                evidence=[
                    "Powerful knee lift"
                ],
                explanation=(
                    "Strong knee drive contributes "
                    "to efficient forward propulsion."
                ),
                recommendation=(
                    "Maintain this lower-body rhythm."
                ),
                priority=1,
            )

        if knee.consistency < 60:
            self.add_finding(
                id="LB003",
                title="Declining Knee Drive",
                domain=Domain.LOWER_BODY,
                confidence=0.88,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Knee drive deteriorated"
                ],
                explanation=(
                    "Lower-body power reduced "
                    "during the run."
                ),
                recommendation=(
                    "Increase endurance work "
                    "to maintain propulsion."
                ),
                priority=4,
            )

        if knee.score > 75 and knee.consistency > 85:
            self.add_finding(
                id="LB004",
                title="Consistent Lower Body",
                domain=Domain.LOWER_BODY,
                confidence=0.93,
                severity=Severity.STRONG,
                evidence=[
                    "Stable knee mechanics"
                ],
                explanation=(
                    "Lower-body mechanics remained "
                    "consistent across the run."
                ),
                recommendation=(
                    "Continue maintaining this rhythm."
                ),
                priority=1,
            )

        if knee.score < 70 and knee.consistency < 70:
            self.add_finding(
                id="LB005",
                title="Lower Body Fatigue",
                domain=Domain.LOWER_BODY,
                confidence=0.89,
                severity=Severity.HIGH_PRIORITY,
                evidence=[
                    "Reduced knee drive",
                    "Poor consistency"
                ],
                explanation=(
                    "The lower body appears to fatigue "
                    "as the run progresses."
                ),
                recommendation=(
                    "Improve endurance and strength "
                    "to sustain propulsion."
                ),
                priority=5,
            )

        if knee.score > 80 and knee.consistency < 55:
            self.add_finding(
                id="LB006",
                title="Power Without Consistency",
                domain=Domain.LOWER_BODY,
                confidence=0.86,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Good peak knee drive",
                    "Large variation"
                ],
                explanation=(
                    "Strong movement exists but is "
                    "not sustained consistently."
                ),
                recommendation=(
                    "Focus on maintaining form "
                    "throughout the entire run."
                ),
                priority=3,
            )

        if knee.score < 55:
            self.add_finding(
                id="LB007",
                title="Reduced Propulsion",
                domain=Domain.LOWER_BODY,
                confidence=0.87,
                severity=Severity.HIGH_PRIORITY,
                evidence=[
                    "Limited knee lift"
                ],
                explanation=(
                    "Forward propulsion is likely "
                    "being limited by insufficient "
                    "leg drive."
                ),
                recommendation=(
                    "Practice running drills that "
                    "emphasize knee lift."
                ),
                priority=4,
            )

        if knee.score > 92 and knee.consistency > 90:
            self.add_finding(
                id="LB008",
                title="Elite Lower Body Mechanics",
                domain=Domain.LOWER_BODY,
                confidence=0.98,
                severity=Severity.ELITE,
                evidence=[
                    "Powerful propulsion",
                    "Exceptional consistency"
                ],
                explanation=(
                    "Lower-body mechanics are at "
                    "an elite level."
                ),
                recommendation=(
                    "Maintain current running form."
                ),
                priority=1,
            )

    def whole_body_rules(self):
        shoulder = self.metric(MetricName.SHOULDER_RELAXATION)
        arm = self.metric(MetricName.ARM_SWING)
        crossover = self.metric(MetricName.ARM_CROSSOVER)
        trunk = self.metric(MetricName.TRUNK_LEAN)
        knee = self.metric(MetricName.KNEE_DRIVE)

        if shoulder.score > 85 and arm.score > 80 and trunk.score > 80 and knee.score > 80:
            self.add_finding(
                id="WB001",
                title="Excellent Whole-Body Coordination",
                domain=Domain.WHOLE_BODY,
                confidence=0.98,
                severity=Severity.ELITE,
                evidence=[
                    "Upper body efficient",
                    "Lower body efficient",
                    "Stable trunk"
                ],
                explanation=(
                    "Your upper and lower body work "
                    "together efficiently."
                ),
                recommendation=(
                    "Maintain your current mechanics."
                ),
                priority=1,
            )

        if shoulder.score > 85 and arm.score > 80 and knee.score < 60:
            self.add_finding(
                id="WB002",
                title="Lower Body Limitation",
                domain=Domain.WHOLE_BODY,
                confidence=0.92,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Strong upper body",
                    "Weak knee drive"
                ],
                explanation=(
                    "Upper body mechanics are already "
                    "efficient. Future gains will "
                    "come from improving propulsion."
                ),
                recommendation=(
                    "Focus on knee drive and lower "
                    "body power."
                ),
                priority=4,
            )

        if knee.score > 85 and arm.score < 60:
            self.add_finding(
                id="WB003",
                title="Upper Body Limitation",
                domain=Domain.WHOLE_BODY,
                confidence=0.91,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Strong propulsion",
                    "Restricted arm swing"
                ],
                explanation=(
                    "Leg mechanics are strong but "
                    "upper-body rhythm is limiting "
                    "overall efficiency."
                ),
                recommendation=(
                    "Relax the shoulders and improve "
                    "arm drive."
                ),
                priority=4,
            )

        if shoulder.consistency > 85 and arm.consistency > 85 and knee.consistency > 85 and trunk.consistency > 85:
            self.add_finding(
                id="WB004",
                title="Stable Running Pattern",
                domain=Domain.WHOLE_BODY,
                confidence=0.97,
                severity=Severity.ELITE,
                evidence=[
                    "Stable shoulders",
                    "Stable trunk",
                    "Stable lower body"
                ],
                explanation=(
                    "Running mechanics remained "
                    "remarkably consistent."
                ),
                recommendation=(
                    "Maintain this rhythm."
                ),
                priority=1,
            )

        if crossover.score < 60 and knee.score < 60:
            self.add_finding(
                id="WB005",
                title="Whole-Body Compensation",
                domain=Domain.WHOLE_BODY,
                confidence=0.90,
                severity=Severity.HIGH_PRIORITY,
                evidence=[
                    "Poor crossover",
                    "Weak propulsion"
                ],
                explanation=(
                    "Upper and lower body appear "
                    "to compensate for one another "
                    "rather than moving as one unit."
                ),
                recommendation=(
                    "Focus on relaxed arm drive "
                    "while improving knee lift."
                ),
                priority=5,
            )

        if trunk.score > 90 and shoulder.score > 80 and crossover.score > 80:
            self.add_finding(
                id="WB006",
                title="Efficient Running Economy",
                domain=Domain.WHOLE_BODY,
                confidence=0.95,
                severity=Severity.STRONG,
                evidence=[
                    "Efficient trunk",
                    "Relaxed shoulders",
                    "Minimal crossover"
                ],
                explanation=(
                    "Movement patterns indicate "
                    "good running economy with "
                    "minimal wasted motion."
                ),
                recommendation=(
                    "Maintain current mechanics "
                    "during longer efforts."
                ),
                priority=2,
            )

    def fatigue_rules(self):
        shoulder = self.metric(MetricName.SHOULDER_RELAXATION)
        arm = self.metric(MetricName.ARM_SWING)
        trunk = self.metric(MetricName.TRUNK_LEAN)
        knee = self.metric(MetricName.KNEE_DRIVE)

        if shoulder.consistency < 70 and knee.consistency < 70:
            self.add_finding(
                id="F001",
                title="Early Fatigue Signs",
                domain=Domain.FATIGUE,
                confidence=0.91,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Shoulder consistency decreased",
                    "Knee drive deteriorated"
                ],
                explanation=(
                    "Both upper and lower body "
                    "began deteriorating during "
                    "the run."
                ),
                recommendation=(
                    "Increase aerobic endurance "
                    "to delay fatigue."
                ),
                priority=4,
            )

        if arm.consistency < 55 and knee.consistency > 80:
            self.add_finding(
                id="F002",
                title="Upper Body Fatigue",
                domain=Domain.FATIGUE,
                confidence=0.89,
                severity=Severity.DEVELOPING,
                evidence=[
                    "Arm rhythm deteriorated",
                    "Lower body remained stable"
                ],
                explanation=(
                    "Fatigue appears isolated "
                    "to the upper body."
                ),
                recommendation=(
                    "Practice relaxed arm rhythm "
                    "during longer runs."
                ),
                priority=3,
            )

        if shoulder.consistency < 60 and arm.consistency < 60 and trunk.consistency < 60 and knee.consistency < 60:
            self.add_finding(
                id="F003",
                title="Progressive Form Breakdown",
                domain=Domain.FATIGUE,
                confidence=0.97,
                severity=Severity.HIGH_PRIORITY,
                evidence=[
                    "Multiple body segments degraded"
                ],
                explanation=(
                    "Running mechanics progressively "
                    "broke down across the body."
                ),
                recommendation=(
                    "Prioritize endurance development "
                    "before increasing speed."
                ),
                priority=5,
            )

        if shoulder.consistency > 90 and arm.consistency > 90 and trunk.consistency > 90 and knee.consistency > 90:
            self.add_finding(
                id="F004",
                title="Excellent Fatigue Resistance",
                domain=Domain.FATIGUE,
                confidence=0.98,
                severity=Severity.ELITE,
                evidence=[
                    "Technique remained stable",
                    "Minimal degradation"
                ],
                explanation=(
                    "Running mechanics stayed "
                    "consistent throughout "
                    "the assessment."
                ),
                recommendation=(
                    "Maintain your endurance training."
                ),
                priority=1,
            )

    def sensor_fusion_rules(self, wearable_data: Dict):
        """
        Combines CV + Wearables + Previous assessments.
        """
        shoulder = self.metric(MetricName.SHOULDER_RELAXATION)
        arm = self.metric(MetricName.ARM_SWING)
        trunk = self.metric(MetricName.TRUNK_LEAN)
        knee = self.metric(MetricName.KNEE_DRIVE)
        crossover = self.metric(MetricName.ARM_CROSSOVER)

        cadence = wearable_data.get("cadence")
        impact = wearable_data.get("impact")
        smoothness = wearable_data.get("smoothness")
        symmetry = wearable_data.get("symmetry")
        vertical = wearable_data.get("vertical_oscillation")

        if knee.score < 60 and cadence < 165 and impact > 80:
            self.add_finding(
                id="SF001",
                title="High Confidence Overstriding",
                domain=Domain.SENSOR_FUSION,
                confidence=0.98,
                severity=Severity.HIGH_PRIORITY,
                evidence=[
                    "Reduced knee drive",
                    "High impact",
                    "Low cadence"
                ],
                explanation=(
                    "Both video analysis and wearable "
                    "data strongly indicate overstriding."
                ),
                recommendation=(
                    "Increase cadence by 5–10 steps "
                    "per minute and land closer "
                    "under your center of mass."
                ),
                priority=5,
                wearable_supported=True,
                cv_supported=True,
            )

        if trunk.score > 85 and smoothness > 80 and vertical < 40:
            self.add_finding(
                id="SF002",
                title="Efficient Running Economy",
                domain=Domain.SENSOR_FUSION,
                confidence=0.95,
                severity=Severity.ELITE,
                evidence=[
                    "Efficient trunk",
                    "Low bounce",
                    "Smooth running"
                ],
                explanation=(
                    "Movement is efficient with "
                    "minimal wasted vertical motion."
                ),
                recommendation=(
                    "Maintain this running rhythm."
                ),
                priority=1,
                wearable_supported=True,
                cv_supported=True,
            )

        if symmetry < 60:
            self.add_finding(
                id="SF003",
                title="Running Asymmetry",
                domain=Domain.SENSOR_FUSION,
                confidence=0.92,
                severity=Severity.NEEDS_ATTENTION,
                evidence=[
                    "Left-right imbalance"
                ],
                explanation=(
                    "One side contributes more "
                    "than the other during running."
                ),
                recommendation=(
                    "Monitor for fatigue or previous "
                    "injury and include unilateral "
                    "strength exercises."
                ),
                priority=4,
                wearable_supported=True,
                cv_supported=False,
            )

        if shoulder.score < 65 and cadence < 165 and smoothness < 65:
            self.add_finding(
                id="SF004",
                title="Fatigue Affecting Form",
                domain=Domain.SENSOR_FUSION,
                confidence=0.94,
                severity=Severity.HIGH_PRIORITY,
                evidence=[
                    "Shoulder tension",
                    "Cadence drop",
                    "Reduced smoothness"
                ],
                explanation=(
                    "Fatigue appears to be reducing "
                    "both posture and movement quality."
                ),
                recommendation=(
                    "Focus on maintaining relaxed "
                    "shoulders during the final "
                    "portion of the run."
                ),
                priority=5,
                wearable_supported=True,
                cv_supported=True,
            )

# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    print("StrideIQ Coaching Engine")
    print("Waiting for biomechanical inputs...")