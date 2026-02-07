"""
Agent 3: IHC Recommendation Engine
Recommends immunohistochemistry stains based on differential diagnosis
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from .base_agent import BaseAgent
from knowledge_base.ihc_database import (
    IHC_STAIN_CATALOG,
    IHCPanel,
    get_ihc_panel,
    get_stain_info,
)


class IHCRecommendation(BaseModel):
    """IHC stain recommendation"""
    stain_name: str
    priority: str  # "Essential" or "Optional"
    rationale: str
    cost: float
    turnaround_hours: int
    expected_result: str


class IHCRecommendationAgent(BaseAgent):
    """Recommends IHC stains to confirm or refute differential diagnoses"""

    def __init__(self):
        super().__init__("IHC Recommendation Engine")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("Generating IHC recommendations")

        differentials = input_data.get("differentials", [])
        primary_diagnosis = (
            differentials[0].get("diagnosis") if differentials else ""
        )

        recommendations = self._generate_panel(primary_diagnosis, differentials)
        total_cost = sum(rec.cost for rec in recommendations)

        return {
            "diagnosis": primary_diagnosis,
            "recommended_panel": [rec.dict() for rec in recommendations],
            "total_cost": total_cost,
            "essential_stains": [rec.stain_name for rec in recommendations if rec.priority == "Essential"],
            "optional_stains": [rec.stain_name for rec in recommendations if rec.priority == "Optional"],
        }

    def _generate_panel(
        self,
        primary_diagnosis: str,
        differentials: List[Dict[str, Any]],
    ) -> List[IHCRecommendation]:
        panel: List[IHCRecommendation] = []

        # Try to load a structured panel first
        structured_panel: IHCPanel | None = get_ihc_panel(primary_diagnosis)
        if structured_panel:
            for stain in structured_panel.essential_stains:
                stain_info = get_stain_info(stain)
                panel.append(
                    IHCRecommendation(
                        stain_name=stain,
                        priority="Essential",
                        rationale=structured_panel.rationale,
                        cost=stain_info.cost if stain_info else 0.0,
                        turnaround_hours=stain_info.turnaround_hours if stain_info else 24,
                        expected_result=stain_info.interpretation.get(primary_diagnosis, "Refer to manual"),
                    )
                )
            for stain in structured_panel.optional_stains:
                stain_info = get_stain_info(stain)
                panel.append(
                    IHCRecommendation(
                        stain_name=stain,
                        priority="Optional",
                        rationale="Provides supportive or prognostic information",
                        cost=stain_info.cost if stain_info else 0.0,
                        turnaround_hours=stain_info.turnaround_hours if stain_info else 24,
                        expected_result=stain_info.interpretation.get(primary_diagnosis, "Refer to manual"),
                    )
                )
            return panel

        # Fallback: basic epithelial marker set
        default_stains = ["Pan-cytokeratin", "EMA", "S100"]
        for stain in default_stains:
            stain_info = IHC_STAIN_CATALOG.get(stain)
            panel.append(
                IHCRecommendation(
                    stain_name=stain,
                    priority="Essential" if stain == "Pan-cytokeratin" else "Optional",
                    rationale="Baseline panel for undifferentiated tumors",
                    cost=stain_info.cost if stain_info else 70.0,
                    turnaround_hours=stain_info.turnaround_hours if stain_info else 24,
                    expected_result="Dependent on lineage",
                )
            )

        return panel
