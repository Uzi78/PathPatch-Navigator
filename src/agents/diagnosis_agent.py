"""
Agent 1: Differential Diagnosis Coordinator
Wraps the MedGemma inference pipeline
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from PIL import Image

from .base_agent import BaseAgent
from inference.differential_diagnosis_agent_8gb import DifferentialDiagnosisAgent


class DiagnosisAgent(BaseAgent):
    """Coordinates differential diagnosis generation"""

    def __init__(self, inference_agent: DifferentialDiagnosisAgent):
        super().__init__("Diagnosis Coordinator")
        self._inference_agent = inference_agent

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("Starting differential diagnosis generation")

        patches: List[Image.Image] = input_data.get("patches", [])
        clinical_context: Dict[str, Any] = input_data.get("clinical_context", {})
        max_diagnoses: int = input_data.get("max_diagnoses", 5)

        if not patches:
            return {
                "differentials": [],
                "recommended_next_steps": "Provide at least one patch for analysis.",
                "teaching_point": "Patch selection is critical for multi-region reasoning."
            }

        result = await asyncio.to_thread(
            self._inference_agent.generate_differential,
            patches,
            clinical_context,
            max_diagnoses,
        )

        self.log_execution(
            f"Generated {len(result.get('differentials', []))} differential diagnoses"
        )
        return result
