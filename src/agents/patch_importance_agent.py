"""
Agent 2: Patch Importance Analyzer
Wraps MedGemma patch-level reasoning
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from PIL import Image

from .base_agent import BaseAgent
from inference.patch_importance_analyzer_8gb import PatchImportanceAnalyzer


class PatchImportanceAgent(BaseAgent):
    """Ranks patches by diagnostic importance"""

    def __init__(self, analyzer: PatchImportanceAnalyzer):
        super().__init__("Patch Importance Analyzer")
        self._analyzer = analyzer

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log_execution("Analyzing patch importance")

        patches: List[Image.Image] = input_data.get("patches", [])
        differentials: List[Dict[str, Any]] = input_data.get("differentials", [])

        if not patches:
            return {
                "rankings": [],
                "most_important_patch": None,
                "summary": "No patches supplied for ranking."
            }

        raw_rankings = await asyncio.to_thread(
            self._analyzer.analyze_importance,
            patches,
            differentials,
        )

        # Sort high-to-low and craft a teaching summary
        rankings = sorted(
            raw_rankings,
            key=lambda record: record.get("importance_score", 0),
            reverse=True,
        )

        summary = "No diagnostic patches identified."
        if rankings:
            top = rankings[0]
            summary = (
                f"Patch {top['patch_id']} is most critical (score: {top['importance_score']}/10). "
                f"{top.get('justification', '').strip()}"
            ).strip()

        return {
            "rankings": rankings,
            "most_important_patch": rankings[0]["patch_id"] if rankings else None,
            "summary": summary,
        }
