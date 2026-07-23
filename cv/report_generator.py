"""
StrideIQ

report_generator.py

Generates the final AI running assessment.

Input

Scores
Findings
Predictions
FeatureVector

↓

Report object

↓

PDF / JSON / Dashboard
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict

from cv.coaching_engine import Finding


# ============================================================
# Report Sections
# ============================================================


@dataclass
class ReportSection:

    title: str

    content: str


# ============================================================
# Final Report
# ============================================================


@dataclass
class RunningReport:

    created_at: datetime

    video_path: str

    overall_score: float

    scores: Dict[str, float]

    findings: List[Finding]

    strengths: List[str]

    priorities: List[str]

    summary: str

    recommendations: List[str]

    feature_snapshot: Dict[str, float]

    sections: List[ReportSection] = field(default_factory=list)


# ============================================================
# Generator
# ============================================================


class ReportGenerator:

    """
    Produces a structured report
    from all CV outputs.
    """

    def generate(

        self,

        scores,

        findings,

        predictions,

        feature_vector,

        video_path,

    ):

        overall = self._overall_score(scores)

        strengths = self._strengths(scores)

        priorities = self._priorities(findings)

        recommendations = self._recommendations(findings)

        summary = self._summary(

            overall,

            strengths,

            priorities,

        )

        report = RunningReport(

            created_at=datetime.now(),

            video_path=video_path,

            overall_score=overall,

            scores=scores,

            findings=findings,

            strengths=strengths,

            priorities=priorities,

            summary=summary,

            recommendations=recommendations,

            feature_snapshot=feature_vector.as_dict(),

        )

        report.sections = self._build_sections(report)

        return report
        # --------------------------------------------------------
    # Overall Score
    # --------------------------------------------------------

    def _overall_score(self, scores):

        if len(scores) == 0:
            return 0.0

        return round(

            sum(scores.values()) / len(scores),

            1

        )

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    def _strengths(self, scores):

        strengths = []

        ordered = sorted(

            scores.items(),

            key=lambda x: x[1],

            reverse=True,

        )

        for domain, score in ordered[:3]:

            strengths.append(

                f"{domain.replace('_', ' ').title()} ({score:.1f}/100)"

            )

        return strengths

    # --------------------------------------------------------
    # Priorities
    # --------------------------------------------------------

    def _priorities(self, findings):

        ordered = sorted(

            findings,

            key=lambda f: (

                f.priority,

                -f.confidence,

            ),

        )

        priorities = []

        for finding in ordered[:5]:

            priorities.append(

                finding.title

            )

        return priorities

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    def _recommendations(self, findings):

        recommendations = []

        seen = set()

        ordered = sorted(

            findings,

            key=lambda f: (

                f.priority,

                -f.confidence,

            ),

        )

        for finding in ordered:

            if finding.recommendation in seen:
                continue

            seen.add(

                finding.recommendation

            )

            recommendations.append(

                finding.recommendation

            )

        return recommendations

    # --------------------------------------------------------
    # Executive Summary
    # --------------------------------------------------------

    def _summary(

        self,

        overall,

        strengths,

        priorities,

    ):

        text = []

        text.append(

            f"Overall running quality score: {overall:.1f}/100."

        )

        if strengths:

            text.append(

                "Strongest areas include "

                + ", ".join(strengths)

                + "."

            )

        if priorities:

            text.append(

                "Highest coaching priorities are "

                + ", ".join(priorities)

                + "."

            )

        text.append(

            "Focus on these priorities during future training sessions."

        )

        return " ".join(text)
        # --------------------------------------------------------
    # Build Report Sections
    # --------------------------------------------------------

    def _build_sections(self, report):

        sections = []

        # =====================================================
        # Executive Summary
        # =====================================================

        sections.append(

            ReportSection(

                title="Executive Summary",

                content=report.summary,

            )

        )

        # =====================================================
        # Overall Performance
        # =====================================================

        overall_text = []

        overall_text.append(

            f"Overall Running Score: {report.overall_score:.1f}/100"

        )

        overall_text.append("")

        for domain, score in sorted(report.scores.items()):

            overall_text.append(

                f"{domain.replace('_',' ').title()}: "

                f"{score:.1f}/100"

            )

        sections.append(

            ReportSection(

                title="Overall Performance",

                content="\n".join(overall_text),

            )

        )

        # =====================================================
        # Strengths
        # =====================================================

        strengths = []

        if len(report.strengths) == 0:

            strengths.append(

                "No clear strengths identified."

            )

        else:

            for s in report.strengths:

                strengths.append(f"• {s}")

        sections.append(

            ReportSection(

                title="Strengths",

                content="\n".join(strengths),

            )

        )

        # =====================================================
        # Priority Findings
        # =====================================================

        finding_text = []

        if len(report.findings) == 0:

            finding_text.append(

                "No significant biomechanical issues detected."

            )

        else:

            ordered = sorted(

                report.findings,

                key=lambda f: (

                    f.priority,

                    -f.confidence,

                ),

            )

            for finding in ordered:

                finding_text.append(

                    f"{finding.title}"

                )

                finding_text.append(

                    f"Severity: {finding.severity.name}"

                )

                finding_text.append(

                    f"Confidence: {finding.confidence:.2f}"

                )

                finding_text.append(

                    finding.explanation

                )

                finding_text.append("")

        sections.append(

            ReportSection(

                title="Priority Findings",

                content="\n".join(finding_text),

            )

        )

        # =====================================================
        # Recommended Drills
        # =====================================================

        drills = []

        if len(report.recommendations) == 0:

            drills.append(

                "No corrective drills recommended."

            )

        else:

            for rec in report.recommendations:

                drills.append(

                    f"• {rec}"

                )

        sections.append(

            ReportSection(

                title="Recommended Drills",

                content="\n".join(drills),

            )

        )

        # =====================================================
        # Feature Snapshot
        # =====================================================

        snapshot = []

        for key, value in sorted(

            report.feature_snapshot.items()

        ):

            snapshot.append(

                f"{key}: {value:.3f}"

            )

        sections.append(

            ReportSection(

                title="Feature Snapshot",

                content="\n".join(snapshot),

            )

        )

        return sections 
        # --------------------------------------------------------
    # Save Report
    # --------------------------------------------------------

    def save(

        self,

        report,

        output_path,

    ):

        """
        Saves report as a nicely formatted text file.

        PDF export can be added later without changing
        the rest of the pipeline.
        """

        from pathlib import Path

        output_path = Path(output_path)

        output_path.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        with open(

            output_path,

            "w",

            encoding="utf-8",

        ) as f:

            f.write("=" * 70 + "\n")
            f.write("StrideIQ AI Running Assessment\n")
            f.write("=" * 70 + "\n\n")

            f.write(
                f"Generated: {report.created_at}\n"
            )

            f.write(
                f"Video: {report.video_path}\n\n"
            )

            f.write(
                f"Overall Score: {report.overall_score:.1f}/100\n\n"
            )

            for section in report.sections:

                f.write("-" * 70 + "\n")

                f.write(section.title + "\n")

                f.write("-" * 70 + "\n\n")

                f.write(section.content)

                f.write("\n\n")

        print(

            f"[StrideIQ] Report saved to "

            f"{output_path}"

        )
            # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    def save_json(

        self,

        report,

        output_path,

    ):

        import json

        from pathlib import Path

        output_path = Path(output_path)

        output_path.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        data = {

            "created_at": str(report.created_at),

            "video_path": report.video_path,

            "overall_score": report.overall_score,

            "scores": report.scores,

            "strengths": report.strengths,

            "priorities": report.priorities,

            "summary": report.summary,

            "recommendations": report.recommendations,  

            "feature_snapshot": report.feature_snapshot,

        }

        with open(

            output_path,

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                data,

                f,

                indent=4,

            ) 