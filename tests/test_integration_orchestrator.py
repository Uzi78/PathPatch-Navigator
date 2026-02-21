import asyncio
import sys
from pathlib import Path
from PIL import Image

# Ensure `src` is on sys.path when running the test directly with `python`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.diagnosis_agent import DiagnosisAgent
from agents.patch_importance_agent import PatchImportanceAgent
from agents.ihc_recommendation_agent import IHCRecommendationAgent
from agents.cost_calculator_agent import CostCalculatorAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.orchestrator import AgentOrchestrator


class DummyDiagnosisInference:
    def generate_differential(self, patches, clinical_context, max_diagnoses):
        return {
            "differentials": [
                {
                    "diagnosis": "TestDx",
                    "confidence": 95,
                    "supporting_features": ["feature1", "feature2"],
                    "teaching_point": "Example teaching point",
                    "recommended_next_steps": "None",
                }
            ]
        }


class DummyPatchAnalyzer:
    def analyze_importance(self, patches, differentials):
        return [{"patch_id": 1, "importance_score": 9.0, "justification": "Obvious"}]


async def test_orchestrator_integration():
    # Create dummy inference wrappers
    diagnosis_inference = DummyDiagnosisInference()
    patch_analyzer = DummyPatchAnalyzer()

    # Wrap in agent classes
    diagnosis_agent = DiagnosisAgent(diagnosis_inference)
    patch_agent = PatchImportanceAgent(patch_analyzer)
    ihc_agent = IHCRecommendationAgent()
    cost_agent = CostCalculatorAgent()
    report_agent = ReportGeneratorAgent()

    orchestrator = AgentOrchestrator(
        diagnosis_agent=diagnosis_agent,
        patch_agent=patch_agent,
        ihc_agent=ihc_agent,
        cost_agent=cost_agent,
        report_agent=report_agent,
    )

    # Create small synthetic patches
    patches = [Image.new("RGB", (32, 32), color=(255, 255, 255)) for _ in range(3)]

    payload = {
        "patches": patches,
        "patch_metadata": [{"patch_id": i + 1} for i in range(len(patches))],
        "clinical_context": {"site": "test"},
        "slide_name": "test-slide",
        "max_diagnoses": 3,
    }

    result = await orchestrator.execute_workflow(payload)

    assert result.get("status") == "success"
    assert "diagnosis" in result
    assert "patch_analysis" in result
    assert "ihc_recommendations" in result
    assert "cost_analysis" in result
    assert "pathology_report" in result


if __name__ == "__main__":
    # Allow running the integration check directly without pytest installed
    asyncio.run(test_orchestrator_integration())
    print("Integration smoke test passed")
