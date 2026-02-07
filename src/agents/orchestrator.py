"""
Agent Orchestrator
Coordinates the execution of all agents in dependency order
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from loguru import logger

from .diagnosis_agent import DiagnosisAgent
from .patch_importance_agent import PatchImportanceAgent
from .ihc_recommendation_agent import IHCRecommendationAgent
from .cost_calculator_agent import CostCalculatorAgent
from .report_generator_agent import ReportGeneratorAgent
from utils.json_formatter import PathPatchJSONFormatter


class AgentOrchestrator:
    """Orchestrates the PathPatch agentic workflow"""

    def __init__(
        self,
        *,
        diagnosis_agent: DiagnosisAgent,
        patch_agent: PatchImportanceAgent,
        ihc_agent: IHCRecommendationAgent,
        cost_agent: CostCalculatorAgent,
        report_agent: ReportGeneratorAgent,
        json_formatter: Optional[PathPatchJSONFormatter] = None,
    ):
        self.diagnosis_agent = diagnosis_agent
        self.patch_agent = patch_agent
        self.ihc_agent = ihc_agent
        self.cost_agent = cost_agent
        self.report_agent = report_agent
        self.json_formatter = json_formatter

    async def execute_workflow(self, case_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full multi-agent workflow"""
        logger.info("=" * 80)
        logger.info("STARTING AGENTIC WORKFLOW")
        logger.info("=" * 80)

        patches = case_input.get("patches", [])
        patch_metadata = case_input.get("patch_metadata", [])
        clinical_context = case_input.get("clinical_context", {})
        slide_name = case_input.get("slide_name", "unknown-slide")
        max_diagnoses = case_input.get("max_diagnoses", 5)

        workflow_result: Dict[str, Any] = {
            "status": "success",
            "case_input": {
                "slide_name": slide_name,
                "clinical_context": clinical_context,
                "patch_count": len(patches),
            },
            "agents_executed": [],
        }

        try:
            # Agent 1: Diagnosis
            logger.info("[1/5] Diagnosis Agent")
            diagnosis_output = await self.diagnosis_agent.execute(
                {
                    "patches": patches,
                    "clinical_context": clinical_context,
                    "max_diagnoses": max_diagnoses,
                }
            )
            workflow_result["diagnosis"] = diagnosis_output
            workflow_result["agents_executed"].append("DiagnosisAgent")

            # Agent 2: Patch importance
            logger.info("[2/5] Patch Importance Agent")
            patch_output = await self.patch_agent.execute(
                {"patches": patches, "differentials": diagnosis_output.get("differentials", [])}
            )
            workflow_result["patch_analysis"] = patch_output
            workflow_result["agents_executed"].append("PatchImportanceAgent")

            # Agent 3: IHC recommendations
            logger.info("[3/5] IHC Recommendation Agent")
            ihc_output = await self.ihc_agent.execute(
                {"differentials": diagnosis_output.get("differentials", [])}
            )
            workflow_result["ihc_recommendations"] = ihc_output
            workflow_result["agents_executed"].append("IHCRecommendationAgent")

            # Agent 4: Cost calculator
            logger.info("[4/5] Cost Calculator Agent")
            cost_output = await self.cost_agent.execute(
                {"recommended_panel": ihc_output.get("recommended_panel", [])}
            )
            workflow_result["cost_analysis"] = cost_output
            workflow_result["agents_executed"].append("CostCalculatorAgent")

            # Agent 5: Report generator
            logger.info("[5/5] Report Generator Agent")
            report_output = await self.report_agent.execute(
                {
                    "clinical_context": clinical_context,
                    "differentials": diagnosis_output.get("differentials", []),
                    "ihc_panel": ihc_output.get("recommended_panel", []),
                    "patch_analysis": patch_output,
                    "cost_analysis": cost_output,
                }
            )
            workflow_result["pathology_report"] = report_output
            workflow_result["agents_executed"].append("ReportGeneratorAgent")

            # Optional formatting for downstream apps
            if self.json_formatter:
                formatted_output = self.json_formatter.format_results(
                    slide_name=slide_name,
                    clinical_context=clinical_context,
                    patches_metadata=patch_metadata,
                    differentials=diagnosis_output.get("differentials", []),
                    patch_importance=patch_output.get("rankings", []),
                    recommended_next_steps=diagnosis_output.get("recommended_next_steps", ""),
                    teaching_point=diagnosis_output.get("teaching_point", ""),
                    ihc_recommendations=ihc_output,
                    cost_analysis=cost_output,
                    report=report_output.get("report", {}),
                )
                output_path = self.json_formatter.save_json(formatted_output)
                workflow_result["formatted_output"] = formatted_output
                workflow_result["output_file"] = output_path

            logger.success("Workflow completed successfully")
            return workflow_result

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Workflow failed: {exc}")
            workflow_result["status"] = "error"
            workflow_result["error"] = str(exc)
            return workflow_result
