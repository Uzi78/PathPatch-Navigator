"""
Full pipeline integration test
Tests the complete workflow from WSI to JSON output
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing.wsi_patch_extractor import WSIPatchExtractor
from inference.differential_diagnosis_agent_8gb import DifferentialDiagnosisAgent
from inference.patch_importance_analyzer_8gb import PatchImportanceAnalyzer
from utils.json_formatter import PathPatchJSONFormatter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_full_pipeline():
    """Test complete pipeline"""
    
    logger.info("\n" + "="*80)
    logger.info("FULL PIPELINE INTEGRATION TEST")
    logger.info("="*80)
    
    # Paths
    MODEL_PATH = str(Path.home() / "pathpatch-navigator/models/medgemma-1.5-4b-it")
    WSI_PATH = str(Path.home() / "pathpatch-navigator/data/wsi_slides/CMU-1.svs")
    
    # Step 1: Extract patches
    logger.info("\n[1/4] Extracting patches from WSI...")
    extractor = WSIPatchExtractor(patch_size=256, output_dir="./test_output/patches")
    
    patch_result = extractor.process_slide(
        slide_path=WSI_PATH,
        num_patches=3,
        save_patches=True
    )
    
    patches = patch_result['patches']
    patches_metadata = patch_result['metadata']
    
    logger.info(f"✓ Extracted {len(patches)} patches")
    
    # Step 2: Generate differential diagnoses
    logger.info("\n[2/4] Generating differential diagnoses...")
    
    diagnosis_agent = DifferentialDiagnosisAgent(
        model_path=MODEL_PATH,
        device="cuda",
        use_4bit=True
    )
    
    clinical_context = {
        'age': '55',
        'gender': 'Female',
        'site': 'Breast, left',
        'specimen_type': 'Core needle biopsy',
        'stain': 'H&E'
    }
    
    diff_result = diagnosis_agent.generate_differential(
        patches=patches,
        clinical_context=clinical_context,
        max_diagnoses=5
    )
    
    differentials = diff_result['differentials']
    logger.info(f"✓ Generated {len(differentials)} diagnoses")
    
    # Step 3: Rank patch importance
    logger.info("\n[3/4] Ranking patch importance...")
    
    importance_analyzer = PatchImportanceAnalyzer(
        model_path=MODEL_PATH,
        device="cuda",
        use_4bit=True
    )
    
    importance_result = importance_analyzer.analyze_importance(
        patches=patches,
        differentials=differentials
    )
    
    logger.info(f"✓ Ranked {len(importance_result)} patches")
    
    # Step 4: Generate JSON output
    logger.info("\n[4/4] Generating structured JSON output...")
    
    formatter = PathPatchJSONFormatter(output_dir="./test_output")
    
    structured_output = formatter.format_results(
        slide_name="CMU-1",
        clinical_context=clinical_context,
        patches_metadata=patches_metadata,
        differentials=differentials,
        patch_importance=importance_result,
        recommended_next_steps=diff_result.get('recommended_next_steps', ''),
        teaching_point=diff_result.get('teaching_point', '')
    )
    
    # Validate
    is_valid = formatter.validate_json_schema(structured_output)
    
    if not is_valid:
        logger.error("JSON validation failed!")
        return False
    
    # Save
    output_file = formatter.save_json(structured_output, filename="pipeline_test_results.json")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"✓ All steps completed successfully")
    logger.info(f"✓ Results saved to: {output_file}")
    logger.info(f"✓ Top diagnosis: {differentials[0]['diagnosis']} ({differentials[0]['confidence']}%)")
    logger.info("="*80)
    
    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)