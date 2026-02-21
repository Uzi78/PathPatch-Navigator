"""Lightweight smoke runner for the full orchestrator without heavy models.

Creates synthetic patches, uses dummy inference backends, runs the
agent orchestrator, and saves formatted JSON to ./test_output.

Run with: python3 scripts/run_smoke_local.py
"""

import sys
import asyncio
from pathlib import Path
from PIL import Image

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.json_formatter import PathPatchJSONFormatter
from agents.diagnosis_agent import DiagnosisAgent
from agents.patch_importance_agent import PatchImportanceAgent
from agents.ihc_recommendation_agent import IHCRecommendationAgent
from agents.cost_calculator_agent import CostCalculatorAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.orchestrator import AgentOrchestrator


class DummyDiagnosisInference:
    def generate_differential(self, patches, clinical_context, max_diagnoses):
        # Return a simple deterministic differential
        return {
            "differentials": [
                {
                    "rank": 1,
                    "diagnosis": "Benign Example Lesion",
                    "confidence": 90,
                    "supporting_features": ["well-circumscribed", "uniform nuclei"],
                    "most_diagnostic_patch": 1,
                    "teaching_point": "Example teaching point",
                }
            ],
            "recommended_next_steps": "No further workup",
            "teaching_point": "Educational note"
        }


class DummyPatchAnalyzer:
    def analyze_importance(self, patches, differentials):
        return [
            {"patch_id": i + 1, "importance_score": float(10 - i), "justification": "Synthetic"}
            for i, _ in enumerate(patches)
        ]


async def main():
    # Prepare synthetic patches
    num_patches = 3
    patches = [Image.new("RGB", (256, 256), color=(200, 200, 200)) for _ in range(num_patches)]
    patches_metadata = [{"patch_id": i + 1, "patch_size": 256} for i in range(num_patches)]

    clinical_context = {
        "age": "55",
        "gender": "Female",
        "site": "Test Site",
        "specimen_type": "Biopsy",
        "stain": "H&E",
    }

    # Build dummy inference wrappers and agents
    diagnosis_inference = DummyDiagnosisInference()
    patch_analyzer = DummyPatchAnalyzer()

    diagnosis_agent = DiagnosisAgent(diagnosis_inference)
    patch_agent = PatchImportanceAgent(patch_analyzer)
    ihc_agent = IHCRecommendationAgent()
    cost_agent = CostCalculatorAgent()
    report_agent = ReportGeneratorAgent()

    json_formatter = PathPatchJSONFormatter(output_dir="./test_output")

    orchestrator = AgentOrchestrator(
        diagnosis_agent=diagnosis_agent,
        patch_agent=patch_agent,
        ihc_agent=ihc_agent,
        cost_agent=cost_agent,
        report_agent=report_agent,
        json_formatter=json_formatter,
    )

    payload = {
        "patches": patches,
        "patch_metadata": patches_metadata,
        "clinical_context": clinical_context,
        "slide_name": "local-smoke-test",
        "max_diagnoses": 3,
    }

    print("Running orchestrator (lightweight dummy run)...")
    result = await orchestrator.execute_workflow(payload)

    print("Workflow status:", result.get("status"))
    if result.get("status") == "success":
        formatted = result.get("formatted_output")
        out_file = result.get("output_file")
        print("Saved formatted JSON to:", out_file)
    else:
        print("Workflow failed:", result.get("error"))


if __name__ == "__main__":
    asyncio.run(main())
