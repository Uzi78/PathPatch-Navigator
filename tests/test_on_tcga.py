"""
Test your diagnosis agent on downloaded TCGA slides
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing.wsi_patch_extractor import WSIPatchExtractor
from inference.differential_diagnosis_agent_8gb import DifferentialDiagnosisAgent
from inference.patch_importance_analyzer_8gb import PatchImportanceAnalyzer
from utils.json_formatter import PathPatchJSONFormatter

import asyncio
import torch

from agents.diagnosis_agent import DiagnosisAgent
from agents.patch_importance_agent import PatchImportanceAgent
from agents.ihc_recommendation_agent import IHCRecommendationAgent
from agents.cost_calculator_agent import CostCalculatorAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.orchestrator import AgentOrchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tcga_slides():
    """Test on all TCGA slides"""
    
    # Paths
    TCGA_DIR = Path.home() / "pathpatch-navigator/data/raw/tcga"
    MODEL_PATH = Path.home() / "pathpatch-navigator/models/medgemma-1.5-4b-it"

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize heavy inference backends once
    logger.info("Initializing inference backends (this may take a while)...")
    diagnosis_inf = DifferentialDiagnosisAgent(
        model_path=str(MODEL_PATH), device=device, use_4bit=(device == "cuda")
    )
    importance_inf = PatchImportanceAnalyzer(
        model_path=str(MODEL_PATH), device=device, use_4bit=(device == "cuda")
    )

    # Wrap in agent classes
    diagnosis_agent = DiagnosisAgent(diagnosis_inf)
    patch_agent = PatchImportanceAgent(importance_inf)
    ihc_agent = IHCRecommendationAgent()
    cost_agent = CostCalculatorAgent()
    report_agent = ReportGeneratorAgent()

    json_formatter = PathPatchJSONFormatter(output_dir="./tcga_test_output")

    orchestrator = AgentOrchestrator(
        diagnosis_agent=diagnosis_agent,
        patch_agent=patch_agent,
        ihc_agent=ihc_agent,
        cost_agent=cost_agent,
        report_agent=report_agent,
        json_formatter=json_formatter,
    )
    
    # Find TCGA slides
    slides = list(TCGA_DIR.glob("*.svs"))
    logger.info(f"Found {len(slides)} TCGA slides")
    
    if not slides:
        logger.error("No TCGA slides found! Run download_tcga.py first")
        return
    
    # Initialize
    extractor = WSIPatchExtractor(
        patch_size=256,
        tissue_threshold=0.3,
        output_dir="./tcga_test_output"
    )
    
    agent = DifferentialDiagnosisAgent(
        model_path=str(MODEL_PATH),
        device="cuda",
        use_4bit=True
    )
    
    # Test each slide
    results = []
    
    for slide_path in slides:
        logger.info(f"\\n{'='*80}")
        logger.info(f"Testing: {slide_path.name}")
        logger.info(f"{'='*80}")
        
        try:
            # Extract patches
            patch_result = extractor.process_slide(
                slide_path=str(slide_path),
                num_patches=3,
                save_patches=True
            )
            
            if patch_result.get('num_patches_extracted', 0) == 0:
                logger.warning(f"No patches extracted from {slide_path.name}")
                continue

            # Build payload for orchestrator (use extracted patches)
            patches = patch_result.get('patches', [])
            patches_metadata = patch_result.get('metadata', [])

            clinical_context = {
                'age': 'Unknown',
                'gender': 'Female',
                'site': 'Breast',
                'specimen_type': 'Biopsy',
                'stain': 'H&E'
            }

            # Run the full agent orchestrator (async)
            workflow_payload = asyncio.run(
                orchestrator.execute_workflow(
                    {
                        'patches': patches,
                        'patch_metadata': patches_metadata,
                        'clinical_context': clinical_context,
                        'slide_name': slide_path.name,
                        'max_diagnoses': 3,
                    }
                )
            )

            differentials = workflow_payload.get('diagnosis', {}).get('differentials', [])
            if differentials:
                top_diagnosis = differentials[0]
                results.append({
                    'slide': slide_path.name,
                    'diagnosis': top_diagnosis['diagnosis'],
                    'confidence': top_diagnosis['confidence'],
                    'num_patches': patch_result['num_patches_extracted']
                })
                
                logger.info(f"✓ Top diagnosis: {top_diagnosis['diagnosis']} ({top_diagnosis['confidence']}%)")
            
        except Exception as e:
            logger.error(f"✗ Failed on {slide_path.name}: {e}")
            continue
    
    # Summary
    logger.info(f"\\n{'='*80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Slides tested: {len(results)}/{len(slides)}")
    
    for result in results:
        logger.info(f"\\n{result['slide']}:")
        logger.info(f"  Diagnosis: {result['diagnosis']}")
        logger.info(f"  Confidence: {result['confidence']}%")
        logger.info(f"  Patches: {result['num_patches']}")
    
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    test_tcga_slides()