"""
IHC (Immunohistochemistry) Knowledge Base
Contains standard IHC panels for common pathology diagnoses
"""

from typing import Dict, List
from pydantic import BaseModel


class IHCStain(BaseModel):
    """Single IHC stain definition"""
    name: str
    cost: float  # USD
    turnaround_hours: int
    interpretation: Dict[str, str]  # diagnosis -> expected result
    sensitivity: float  # 0-1
    specificity: float  # 0-1


class IHCPanel(BaseModel):
    """Complete IHC panel for a diagnosis"""
    diagnosis: str
    essential_stains: List[str]
    optional_stains: List[str]
    total_cost: float
    rationale: str


# Complete IHC Stain Catalog
IHC_STAIN_CATALOG = {
    "E-cadherin": IHCStain(
        name="E-cadherin",
        cost=85.0,
        turnaround_hours=24,
        interpretation={
            "Invasive Ductal Carcinoma": "Positive (membranous)",
            "Invasive Lobular Carcinoma": "Negative (loss of expression)",
            "Normal Breast Tissue": "Positive (membranous)"
        },
        sensitivity=0.95,
        specificity=0.92
    ),
    
    "ER": IHCStain(
        name="Estrogen Receptor (ER)",
        cost=60.0,
        turnaround_hours=24,
        interpretation={
            "ER-Positive Breast Cancer": "Positive (nuclear, >1%)",
            "ER-Negative Breast Cancer": "Negative (<1%)",
            "Triple-Negative Breast Cancer": "Negative"
        },
        sensitivity=0.88,
        specificity=0.91
    ),
    
    "PR": IHCStain(
        name="Progesterone Receptor (PR)",
        cost=60.0,
        turnaround_hours=24,
        interpretation={
            "PR-Positive Breast Cancer": "Positive (nuclear, >1%)",
            "PR-Negative Breast Cancer": "Negative (<1%)"
        },
        sensitivity=0.82,
        specificity=0.89
    ),
    
    "HER2": IHCStain(
        name="HER2",
        cost=80.0,
        turnaround_hours=48,
        interpretation={
            "HER2-Positive Breast Cancer": "3+ (strong complete membrane)",
            "HER2-Negative Breast Cancer": "0 or 1+",
            "Equivocal": "2+ (requires FISH confirmation)"
        },
        sensitivity=0.86,
        specificity=0.94
    ),
    
    "Ki-67": IHCStain(
        name="Ki-67 Proliferation Index",
        cost=75.0,
        turnaround_hours=24,
        interpretation={
            "Low Grade Tumor": "<10% positive nuclei",
            "Intermediate Grade": "10-20% positive nuclei",
            "High Grade Tumor": ">20% positive nuclei"
        },
        sensitivity=0.79,
        specificity=0.81
    ),
    
    "p120-catenin": IHCStain(
        name="p120 catenin",
        cost=90.0,
        turnaround_hours=24,
        interpretation={
            "Invasive Ductal Carcinoma": "Membranous pattern",
            "Invasive Lobular Carcinoma": "Cytoplasmic pattern"
        },
        sensitivity=0.84,
        specificity=0.87
    ),
    
    "CK5/6": IHCStain(
        name="Cytokeratin 5/6",
        cost=70.0,
        turnaround_hours=24,
        interpretation={
            "Basal-like Breast Cancer": "Positive",
            "Luminal Breast Cancer": "Negative",
            "Myoepithelial Cells": "Positive"
        },
        sensitivity=0.76,
        specificity=0.83
    ),
    
    "p63": IHCStain(
        name="p63",
        cost=68.0,
        turnaround_hours=24,
        interpretation={
            "Myoepithelial Cells": "Positive (nuclear)",
            "Invasive Carcinoma": "Negative",
            "DCIS with preserved myoepithelium": "Positive in outer layer"
        },
        sensitivity=0.92,
        specificity=0.89
    ),
    
    "PSA": IHCStain(
        name="Prostate-Specific Antigen (PSA)",
        cost=65.0,
        turnaround_hours=24,
        interpretation={
            "Prostate Adenocarcinoma": "Positive",
            "High-Grade PIN": "Positive",
            "Benign Prostate": "Positive"
        },
        sensitivity=0.95,
        specificity=0.98
    ),
    
    "AMACR": IHCStain(
        name="AMACR (P504S)",
        cost=75.0,
        turnaround_hours=24,
        interpretation={
            "Prostate Adenocarcinoma": "Positive (cytoplasmic)",
            "High-Grade PIN": "Positive",
            "Benign Prostate Glands": "Negative"
        },
        sensitivity=0.88,
        specificity=0.79
    ),
    
    "p16": IHCStain(
        name="p16",
        cost=72.0,
        turnaround_hours=24,
        interpretation={
            "HPV-Related Cervical Dysplasia": "Diffuse block-positive",
            "Non-HPV Lesions": "Patchy or negative",
            "CIN2+": "Strong diffuse nuclear and cytoplasmic"
        },
        sensitivity=0.91,
        specificity=0.84
    ),
}


# Standard IHC Panels by Diagnosis
STANDARD_IHC_PANELS = {
    "Invasive Ductal Carcinoma": IHCPanel(
        diagnosis="Invasive Ductal Carcinoma",
        essential_stains=["E-cadherin", "ER", "PR", "HER2"],
        optional_stains=["Ki-67"],
        total_cost=285.0,
        rationale="E-cadherin confirms ductal origin. ER/PR/HER2 required for treatment planning per ASCO guidelines."
    ),
    
    "Invasive Lobular Carcinoma": IHCPanel(
        diagnosis="Invasive Lobular Carcinoma",
        essential_stains=["E-cadherin", "ER", "PR", "HER2"],
        optional_stains=["p120-catenin"],
        total_cost=285.0,
        rationale="E-cadherin loss is diagnostic for ILC. ER/PR typically positive in ILC."
    ),
    
    "Triple-Negative Breast Cancer": IHCPanel(
        diagnosis="Triple-Negative Breast Cancer",
        essential_stains=["ER", "PR", "HER2", "CK5/6"],
        optional_stains=["Ki-67", "p63"],
        total_cost=270.0,
        rationale="Confirm triple-negative status. CK5/6 identifies basal-like subtype."
    ),
    
    "Prostate Adenocarcinoma": IHCPanel(
        diagnosis="Prostate Adenocarcinoma",
        essential_stains=["PSA", "AMACR"],
        optional_stains=["p63"],
        total_cost=140.0,
        rationale="AMACR positive in cancer, p63 negative (vs. positive in benign)."
    ),
    
    "Cervical Dysplasia (CIN)": IHCPanel(
        diagnosis="Cervical Intraepithelial Neoplasia",
        essential_stains=["p16"],
        optional_stains=["Ki-67"],
        total_cost=72.0,
        rationale="p16 block positivity confirms HPV-related dysplasia, aids in CIN2/3 diagnosis."
    ),
}


def get_ihc_panel(diagnosis: str) -> IHCPanel:
    """Retrieve standard IHC panel for a diagnosis"""
    return STANDARD_IHC_PANELS.get(diagnosis)


def get_stain_info(stain_name: str) -> IHCStain:
    """Retrieve detailed info for a specific stain"""
    return IHC_STAIN_CATALOG.get(stain_name)


def calculate_panel_cost(stain_list: List[str]) -> float:
    """Calculate total cost for a list of stains"""
    total = 0.0
    for stain in stain_list:
        stain_info = IHC_STAIN_CATALOG.get(stain)
        if stain_info:
            total += stain_info.cost
    return total
