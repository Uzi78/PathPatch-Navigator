"""
Agent 4: Cost & Availability Calculator
Calculates IHC panel costs and checks institutional availability
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from .base_agent import BaseAgent
from knowledge_base.cost_database import CostCalculator, DEFAULT_INSTITUTION


class CostAnalysis(BaseModel):
    """Cost analysis result"""
    total_cost: float
    requires_approval: bool
    approval_threshold: float
    breakdown: Dict[str, float]
    unavailable_stains: List[str]
    estimated_turnaround_hours: int


class CostCalculatorAgent(BaseAgent):
    """Calculates costs and checks stain availability"""

    def __init__(self):
        super().__init__("Cost & Availability Calculator")
        self.cost_calculator = CostCalculator(DEFAULT_INSTITUTION)
        self.available_stains = self._load_inventory()

    def _load_inventory(self) -> List[str]:
        """Simulate institutional stain availability"""
        return [
            "E-cadherin",
            "ER",
            "PR",
            "HER2",
            "Ki-67",
            "PSA",
            "AMACR",
            "p63",
            "p16",
            "CK5/6",
        ]

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("Calculating cost and availability")

        recommended_panel = input_data.get("recommended_panel", [])
        stain_names = [stain.get("stain_name") for stain in recommended_panel]

        if not stain_names:
            return CostAnalysis(
                total_cost=0.0,
                requires_approval=False,
                approval_threshold=self.cost_calculator.institution.resident_approval_limit,
                breakdown={},
                unavailable_stains=[],
                estimated_turnaround_hours=0,
            ).dict()

        cost_result = self.cost_calculator.calculate_panel_cost(stain_names)
        unavailable = [stain for stain in stain_names if stain not in self.available_stains]

        turnaround = max(
            (stain.get("turnaround_hours", 24) for stain in recommended_panel),
            default=24,
        )
        if unavailable:
            turnaround += 72  # send-out penalty

        analysis = CostAnalysis(
            total_cost=cost_result["total_cost"],
            requires_approval=cost_result["requires_approval"],
            approval_threshold=cost_result["approval_threshold"],
            breakdown=cost_result["breakdown"],
            unavailable_stains=unavailable,
            estimated_turnaround_hours=turnaround,
        )

        return analysis.dict()
