"""
Agent 5: Report Template Generator
Creates structured pathology reports following CAP guidelines
"""

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel

from .base_agent import BaseAgent


class PathologyReport(BaseModel):
    """Structured pathology report"""
    report_id: str
    date: str
    specimen_type: str
    clinical_history: str
    gross_description: str
    microscopic_description: str
    immunohistochemistry: str
    diagnosis: str
    comment: str
    synoptic_elements: Dict[str, str]


class ReportGeneratorAgent(BaseAgent):
    """Generates CAP-aligned educational report templates"""

    def __init__(self):
        super().__init__("Report Template Generator")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("Building educational report")
        report = self._build_report(input_data)
        formatted = self._format_report(report)
        return {"report": report.dict(), "formatted_report": formatted}

    def _build_report(self, data: Dict[str, Any]) -> PathologyReport:
        clinical_context = data.get("clinical_context", {})
        differentials: List[Dict[str, Any]] = data.get("differentials", [])
        ihc_panel: List[Dict[str, Any]] = data.get("ihc_panel", [])
        patch_analysis: Dict[str, Any] = data.get("patch_analysis", {})

        primary_dx = differentials[0] if differentials else {}
        microscopic = self._microscopic_description(primary_dx)
        ihc_section = self._ihc_section(ihc_panel)
        comment = self._educational_comment(differentials, ihc_panel)
        synoptic = self._synoptic_elements(primary_dx, patch_analysis)

        return PathologyReport(
            report_id=f"PATH-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            date=datetime.now().strftime("%Y-%m-%d"),
            specimen_type=clinical_context.get("specimen_type", "Unknown"),
            clinical_history=clinical_context.get("clinical_info", "Not provided"),
            gross_description=clinical_context.get("gross_exam", "Not provided"),
            microscopic_description=microscopic,
            immunohistochemistry=ihc_section,
            diagnosis=primary_dx.get("diagnosis", "Pending further analysis"),
            comment=comment,
            synoptic_elements=synoptic,
        )

    def _microscopic_description(self, primary_dx: Dict[str, Any]) -> str:
        features = primary_dx.get("supporting_features", [])
        if not features:
            return "Sections reviewed."
        body = "\n".join(f"- {feature}" for feature in features)
        return f"Sections demonstrate the following features:\n{body}"

    def _ihc_section(self, ihc_panel: List[Dict[str, Any]]) -> str:
        if not ihc_panel:
            return "No immunohistochemistry ordered at this stage."
        essential = [item for item in ihc_panel if item.get("priority") == "Essential"]
        optional = [item for item in ihc_panel if item.get("priority") == "Optional"]
        parts: List[str] = []
        if essential:
            parts.append("ESSENTIAL STAINS:")
            for stain in essential:
                parts.append(
                    f"- {stain['stain_name']}: {stain.get('expected_result', 'Interpret per lab protocol')}\n"
                    f"  Rationale: {stain.get('rationale', 'N/A')}"
                )
        if optional:
            parts.append("\nOPTIONAL STAINS:")
            for stain in optional:
                parts.append(
                    f"- {stain['stain_name']}: {stain.get('expected_result', 'Interpret per lab protocol')}\n"
                    f"  Rationale: {stain.get('rationale', 'N/A')}"
                )
        return "\n".join(parts)

    def _educational_comment(
        self,
        differentials: List[Dict[str, Any]],
        ihc_panel: List[Dict[str, Any]],
    ) -> str:
        if not differentials:
            return "Educational review pending additional material."
        primary = differentials[0]
        comment_lines = ["TEACHING POINT:", primary.get("teaching_point", "Not provided"), ""]
        if len(differentials) > 1:
            comment_lines.append("ALTERNATIVE DIAGNOSES CONSIDERED:")
            for diff in differentials[1:3]:
                comment_lines.append(
                    f"- {diff['diagnosis']} ({diff.get('confidence', 0)}% confidence)"
                )
        if ihc_panel:
            essentials = [stain['stain_name'] for stain in ihc_panel if stain.get('priority') == 'Essential']
            comment_lines.append("")
            comment_lines.append(
                f"RECOMMENDED IHC WORKUP: {', '.join(essentials) if essentials else 'Tailor to morphology'}"
            )
        return "\n".join(line for line in comment_lines if line is not None)

    def _synoptic_elements(
        self,
        primary_dx: Dict[str, Any],
        patch_analysis: Dict[str, Any],
    ) -> Dict[str, str]:
        elements = {
            "Primary Diagnosis": primary_dx.get("diagnosis", "Pending"),
            "Confidence Level": f"{primary_dx.get('confidence', 0)}%",
        }
        grade_hint = primary_dx.get("diagnosis", "")
        if "Grade 2" in grade_hint:
            elements["Histologic Grade"] = "Grade 2 (Moderately differentiated)"
        elif "Grade 3" in grade_hint:
            elements["Histologic Grade"] = "Grade 3 (Poorly differentiated)"
        if patch_analysis.get("most_important_patch") is not None:
            elements["Most Diagnostic Patch"] = (
                f"Patch {patch_analysis['most_important_patch']}"
            )
        return elements

    def _format_report(self, report: PathologyReport) -> str:
        return f"""
=============================================================================
                        SURGICAL PATHOLOGY REPORT
                         (EDUCATIONAL TEMPLATE)
=============================================================================

Report ID: {report.report_id}
Date: {report.date}

SPECIMEN: {report.specimen_type}

CLINICAL HISTORY:
{report.clinical_history}

GROSS DESCRIPTION:
{report.gross_description}

MICROSCOPIC DESCRIPTION:
{report.microscopic_description}

IMMUNOHISTOCHEMISTRY:
{report.immunohistochemistry}

DIAGNOSIS:
{report.diagnosis}

COMMENT:
{report.comment}

SYNOPTIC REPORT:
{self._format_synoptic(report.synoptic_elements)}

=============================================================================
                    FOR EDUCATIONAL USE ONLY
          This report was generated by AI for teaching purposes.
         All findings should be reviewed by an attending pathologist.
=============================================================================
""".strip()

    def _format_synoptic(self, synoptic: Dict[str, str]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in synoptic.items())
