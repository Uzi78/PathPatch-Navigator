"""
Institutional Cost Database
Pricing for IHC stains and institutional limits
"""

from typing import Dict, List
from pydantic import BaseModel


class InstitutionalCosts(BaseModel):
    """Cost configuration for an institution"""
    institution_name: str
    ihc_costs: Dict[str, float]
    resident_approval_limit: float  # Max cost before attending approval needed
    send_out_fee: float  # Additional fee for reference lab


# Example institutional pricing (can be customized per institution)
DEFAULT_INSTITUTION = InstitutionalCosts(
    institution_name="Academic Medical Center",
    ihc_costs={
        "E-cadherin": 85.0,
        "ER": 60.0,
        "PR": 60.0,
        "HER2": 80.0,
        "Ki-67": 75.0,
        "p120-catenin": 90.0,
        "CK5/6": 70.0,
        "p63": 68.0,
        "PSA": 65.0,
        "AMACR": 75.0,
        "p16": 72.0,
    },
    resident_approval_limit=300.0,
    send_out_fee=50.0
)


class CostCalculator:
    """Calculate costs for IHC panels"""

    def __init__(self, institution: InstitutionalCosts = DEFAULT_INSTITUTION):
        self.institution = institution

    def get_stain_cost(self, stain_name: str) -> float:
        """Get cost for a single stain"""
        return self.institution.ihc_costs.get(stain_name, 0.0)

    def calculate_panel_cost(self, stain_list: List[str]) -> Dict[str, any]:
        """Calculate total cost and check approval requirements"""
        breakdown = {stain: self.get_stain_cost(stain) for stain in stain_list}
        total = sum(breakdown.values())

        return {
            "total_cost": total,
            "requires_approval": total > self.institution.resident_approval_limit,
            "breakdown": breakdown,
            "approval_threshold": self.institution.resident_approval_limit
        }
