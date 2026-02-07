"""
JSON Output Formatter
Converts analysis results to structured JSON format
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathPatchJSONFormatter:
    """
    Formats PathPatch Navigator results as structured JSON
    """
    
    def __init__(self, output_dir: str = "./output"):
        """Initialize formatter"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_results(
        self,
        slide_name: str,
        clinical_context: Dict[str, Any],
        patches_metadata: List[Dict[str, Any]],
        differentials: List[Dict[str, Any]],
        patch_importance: List[Dict[str, Any]],
        recommended_next_steps: str = "",
        teaching_point: str = "",
        ihc_recommendations: Optional[Dict[str, Any]] = None,
        cost_analysis: Optional[Dict[str, Any]] = None,
        report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create structured JSON output
        
        Args:
            slide_name: Name of WSI
            clinical_context: Clinical information
            patches_metadata: Patch extraction metadata
            differentials: Differential diagnoses
            patch_importance: Patch importance rankings
            recommended_next_steps: Suggested next actions
            teaching_point: Educational note
        
        Returns:
            Structured dictionary
        """
        
        structured_output = {
            "metadata": {
                "slide_name": slide_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "model": "medgemma-1.5-4b-it"
            },
            
            "clinical_context": clinical_context,
            
            "patches": [
                {
                    "patch_id": p['patch_id'],
                    "coordinates": p.get('coordinates', {}),
                    "patch_size": p.get('patch_size', 256),
                    "magnification": p.get('magnification', 20)
                }
                for p in patches_metadata
            ],
            
            "differential_diagnoses": [
                {
                    "rank": d['rank'],
                    "diagnosis": d['diagnosis'],
                    "confidence_score": d['confidence'],
                    "supporting_features": d.get('supporting_features', []),
                    "most_diagnostic_patch": d.get('most_diagnostic_patch'),
                    "arguments_against": d.get('against_reasoning', '')
                }
                for d in differentials
            ],
            
            "patch_importance_ranking": [
                {
                    "patch_id": p.get('patch_id'),
                    "importance_score": p.get('importance_score'),
                    "rank": idx + 1,
                    "structures_present": p.get('structures_present', []),
                    "critical_features": p.get('critical_features', []),
                    "justification": p.get('justification', '')
                }
                for idx, p in enumerate(patch_importance)
            ],
            "ihc_recommendations": ihc_recommendations or {},
            "cost_analysis": cost_analysis or {},
            "report": report or {},
            
            "recommendations": {
                "next_steps": recommended_next_steps,
                "teaching_point": teaching_point
            },
            
            "summary": {
                "total_patches_analyzed": len(patches_metadata),
                "top_diagnosis": differentials[0]['diagnosis'] if differentials else "None",
                "top_diagnosis_confidence": differentials[0]['confidence'] if differentials else 0,
                "most_important_patch": patch_importance[0].get('patch_id') if patch_importance else None,
                "ihc_panel_size": len((ihc_recommendations or {}).get('recommended_panel', [])),
                "estimated_cost": (cost_analysis or {}).get('total_cost', 0.0)
            }
        }
        
        return structured_output
    
    def save_json(
        self,
        data: Dict[str, Any],
        filename: str = None
    ) -> str:
        """
        Save JSON to file
        
        Args:
            data: Dictionary to save
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slide_name = data['metadata'].get('slide_name', 'unknown')
            filename = f"{slide_name}_{timestamp}_results.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved JSON results to: {output_path}")
        
        return str(output_path)
    
    def validate_json_schema(self, data: Dict[str, Any]) -> bool:
        """
        Validate JSON structure
        
        Args:
            data: Dictionary to validate
        
        Returns:
            True if valid
        """
        required_fields = [
            'metadata',
            'clinical_context',
            'patches',
            'differential_diagnoses',
            'patch_importance_ranking',
            'recommendations',
            'summary'
        ]
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        
        logger.info("✓ JSON structure is valid")
        return True


# Test function
def main():
    """Test JSON formatter"""
    
    # Sample data
    clinical_context = {
        'age': '55',
        'gender': 'Female',
        'site': 'Breast, left',
        'specimen_type': 'Core needle biopsy'
    }
    
    patches_metadata = [
        {'patch_id': 1, 'coordinates': {'x': 1000, 'y': 2000}, 'patch_size': 256},
        {'patch_id': 2, 'coordinates': {'x': 3000, 'y': 4000}, 'patch_size': 256},
        {'patch_id': 3, 'coordinates': {'x': 5000, 'y': 6000}, 'patch_size': 256}
    ]
    
    differentials = [
        {
            'rank': 1,
            'diagnosis': 'Invasive Ductal Carcinoma, Grade 2',
            'confidence': 75,
            'supporting_features': [
                'Infiltrating tumor cells in nests and cords',
                'Moderate nuclear pleomorphism',
                '6 mitoses per 10 HPF'
            ],
            'most_diagnostic_patch': 1,
            'against_reasoning': 'None significant'
        }
    ]
    
    patch_importance = [
        {
            'patch_id': 1,
            'importance_score': 9,
            'structures_present': ['Tumor nests', 'Stromal invasion'],
            'justification': 'Shows classic infiltrative pattern'
        }
    ]
    
    # Create formatter
    formatter = PathPatchJSONFormatter(output_dir="./output")
    
    # Format results
    structured_data = formatter.format_results(
        slide_name="CMU-1",
        clinical_context=clinical_context,
        patches_metadata=patches_metadata,
        differentials=differentials,
        patch_importance=patch_importance,
        recommended_next_steps="Order E-cadherin IHC",
        teaching_point="E-cadherin distinguishes IDC from ILC"
    )
    
    # Validate
    is_valid = formatter.validate_json_schema(structured_data)
    
    if is_valid:
        # Save
        output_path = formatter.save_json(structured_data)
        
        # Print sample
        print("\n" + "="*80)
        print("SAMPLE JSON OUTPUT")
        print("="*80)
        print(json.dumps(structured_data, indent=2)[:1000] + "...")
        print("="*80)
        print(f"\nFull output saved to: {output_path}")


if __name__ == "__main__":
    main()