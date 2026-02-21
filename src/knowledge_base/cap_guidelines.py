"""
CAP (College of American Pathologists) Guidelines
Structured reporting templates and diagnostic criteria
"""

from typing import Dict, List
from pydantic import BaseModel


class DiagnosticCriteria(BaseModel):
    """Criteria for a specific diagnosis"""
    diagnosis: str
    required_features: List[str]
    supporting_features: List[str]
    exclusion_criteria: List[str]
    grading_system: str


CAP_DIAGNOSTIC_CRITERIA = {
    "Invasive Ductal Carcinoma": DiagnosticCriteria(
        diagnosis="Invasive Ductal Carcinoma",
        required_features=[
            "Infiltrative growth pattern disrupting normal architecture",
            "Malignant epithelial cells forming irregular nests/cords",
            "Stromal desmoplastic response"
        ],
        supporting_features=[
            "Nuclear pleomorphism",
            "Increased mitotic activity",
            "Loss of myoepithelial layer",
            "Absence of lobular growth pattern"
        ],
        exclusion_criteria=[
            "Preservation of lobular architecture",
            "No invasion beyond basement membrane (DCIS)"
        ],
        grading_system="Nottingham (Elston-Ellis): Tubule formation + Nuclear grade + Mitotic count"
    ),
    
    "Invasive Lobular Carcinoma": DiagnosticCriteria(
        diagnosis="Invasive Lobular Carcinoma",
        required_features=[
            "Single-file infiltrative pattern",
            "Targetoid growth around normal ducts",
            "Loss of E-cadherin expression (IHC)"
        ],
        supporting_features=[
            "Bland, monotonous tumor cells",
            "Minimal nuclear pleomorphism",
            "Signet ring cells (classic type)",
            "p120-catenin cytoplasmic pattern"
        ],
        exclusion_criteria=[
            "Cohesive nests and cords (more typical of IDC)",
            "Retained E-cadherin expression"
        ],
        grading_system="Nottingham system (though less commonly applied to ILC)"
    ),
    
    "Prostate Adenocarcinoma": DiagnosticCriteria(
        diagnosis="Prostate Adenocarcinoma",
        required_features=[
            "Crowded small glands with minimal intervening stroma",
            "Absence of basal cell layer (p63 negative)",
            "Nuclear enlargement and prominent nucleoli"
        ],
        supporting_features=[
            "AMACR (P504S) positive",
            "PSA positive",
            "Intraluminal blue mucin",
            "Perineural invasion"
        ],
        exclusion_criteria=[
            "Preservation of basal cells (p63 positive)",
            "Normal glandular spacing"
        ],
        grading_system="Gleason Score: Primary + Secondary pattern (Grade Groups 1-5)"
    ),
}


# Nottingham Grading System
NOTTINGHAM_GRADING = {
    "tubule_formation": {
        1: ">75% of tumor forms tubules",
        2: "10-75% tubule formation",
        3: "<10% tubule formation"
    },
    "nuclear_pleomorphism": {
        1: "Small, uniform nuclei",
        2: "Moderate variation in size/shape",
        3: "Marked variation, vesicular chromatin, prominent nucleoli"
    },
    "mitotic_count": {
        1: "0-9 mitoses per 10 HPF",
        2: "10-19 mitoses per 10 HPF",
        3: "≥20 mitoses per 10 HPF"
    },
    "total_score": {
        "3-5": "Grade 1 (Well differentiated)",
        "6-7": "Grade 2 (Moderately differentiated)",
        "8-9": "Grade 3 (Poorly differentiated)"
    }
}


def get_diagnostic_criteria(diagnosis: str) -> DiagnosticCriteria:
    """Get CAP diagnostic criteria for a diagnosis"""
    return CAP_DIAGNOSTIC_CRITERIA.get(diagnosis)


def calculate_nottingham_grade(tubule_score: int, nuclear_score: int, mitotic_score: int) -> str:
    """Calculate Nottingham grade from component scores"""
    total = tubule_score + nuclear_score + mitotic_score
    
    if 3 <= total <= 5:
        return "Grade 1 (Well differentiated)"
    if 6 <= total <= 7:
        return "Grade 2 (Moderately differentiated)"
    if 8 <= total <= 9:
        return "Grade 3 (Poorly differentiated)"
    return "Invalid score"
