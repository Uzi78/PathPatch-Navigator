"""
Differential Diagnosis Agent - Optimized for 8GB VRAM
Uses 4-bit quantization and sequential processing
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from typing import List, Dict, Any
import logging
import re
import json
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifferentialDiagnosisAgent:
    """
    Agent for generating differential diagnoses from histopathology patches
    Optimized for 8GB VRAM GPUs (RTX 3060 Ti)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_4bit: bool = True
    ):
        """
        Initialize the diagnosis agent with 4-bit quantization
        
        Args:
            model_path: Path to MedGemma model
            device: Device to run inference on ('cuda' or 'cpu')
            use_4bit: Whether to use 4-bit quantization (recommended for 8GB VRAM)
        """
        self.device = device
        logger.info(f"Initializing DifferentialDiagnosisAgent on {device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Configure 4-bit quantization
        bnb_config = None
        if use_4bit and device == "cuda":
            logger.info("Using 4-bit quantization for memory efficiency...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        logger.info("Loading model (may take 2-3 minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        logger.info(f"✓ Model loaded successfully!")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"GPU Memory Usage: {allocated:.2f} GB / 8.00 GB")
    
    def generate_differential(
        self,
        patches: List[Image.Image],
        clinical_context: Dict[str, Any],
        max_diagnoses: int = 5
    ) -> Dict[str, Any]:
        """
        Generate differential diagnosis from multiple patches
        
        Args:
            patches: List of PIL Images (3-5 patches)
            clinical_context: Dictionary with clinical information
            max_diagnoses: Maximum number of diagnoses to generate
        
        Returns:
            Dictionary containing ranked differential diagnoses
        """
        logger.info(f"\nGenerating differential from {len(patches)} patches...")
        
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create prompt
        prompt = self._create_prompt(patches, clinical_context, max_diagnoses)
        
        # Prepare inputs
        logger.info("Encoding inputs...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Check memory before generation
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Pre-generation GPU memory: {allocated:.2f} GB")
        
        # Generate response with memory-efficient settings
        logger.info("Generating diagnosis (this may take 45-90 seconds on RTX 3060 Ti)...")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                # Memory optimization: use_cache=False reduces memory usage
                use_cache=True  # Keep True for speed, set False if OOM
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clear cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Parse structured output
        logger.info("Parsing response...")
        structured_output = self._parse_response(response)
        
        logger.info(f"✓ Generated {len(structured_output.get('differentials', []))} diagnoses")
        
        # Memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Post-generation GPU memory: {allocated:.2f} GB")
        
        return structured_output
    
    def _create_prompt(
        self,
        patches: List[Image.Image],
        clinical_context: Dict[str, Any],
        max_diagnoses: int
    ) -> str:
        """Create structured prompt for MedGemma"""
        
        age = clinical_context.get('age', 'Unknown')
        gender = clinical_context.get('gender', 'Unknown')
        site = clinical_context.get('site', 'Unknown')
        specimen_type = clinical_context.get('specimen_type', 'Biopsy')
        stain = clinical_context.get('stain', 'H&E')
        
        prompt = f"""You are an expert pathologist teaching a pathology resident.

CLINICAL CONTEXT:
- Patient Age: {age}
- Gender: {gender}
- Anatomic Site: {site}
- Specimen Type: {specimen_type}
- Stain: {stain}

TASK:
Analyze these {len(patches)} H&E stained patches from the same whole-slide image.
Generate a differential diagnosis with exactly the following structure:

DIFFERENTIAL DIAGNOSIS (Top {max_diagnoses}):

For each diagnosis, provide:
1. Diagnosis name
2. Confidence score (0-100%)
3. Key histologic features supporting this diagnosis
4. Which specific patch number (1-{len(patches)}) shows the most diagnostic features
5. Features that argue against this diagnosis (for teaching)

Format each diagnosis as:

[RANK]. [DIAGNOSIS NAME] - [CONFIDENCE]% confidence
   Key Features:
   - [Feature 1 with patch reference]
   - [Feature 2 with patch reference]
   - [Feature 3 with patch reference]
   Most Diagnostic Patch: Patch [NUMBER]
   Against: [Why this diagnosis might be wrong]

RECOMMENDED NEXT STEPS:
[What additional patches or stains would be helpful]

TEACHING POINT:
[One key learning point for residents]

Begin analysis:
"""
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse MedGemma response into structured format"""
        
        structured = {
            'differentials': [],
            'recommended_next_steps': '',
            'teaching_point': ''
        }
        
        # Extract differential diagnoses
        diagnosis_pattern = r'(\d+)\.\s+([^-]+?)\s+-\s+(\d+)%\s+confidence'
        matches = re.finditer(diagnosis_pattern, response, re.MULTILINE | re.IGNORECASE)
        
        for match in matches:
            rank = int(match.group(1))
            diagnosis_name = match.group(2).strip()
            confidence = int(match.group(3))
            
            # Extract features section
            start_pos = match.end()
            next_match = re.search(r'\n\d+\.', response[start_pos:])
            if next_match:
                end_pos = start_pos + next_match.start()
            else:
                next_section = re.search(r'\n(RECOMMENDED|TEACHING)', response[start_pos:])
                end_pos = start_pos + next_section.start() if next_section else len(response)
            
            diagnosis_text = response[start_pos:end_pos]
            
            # Extract key features
            features = []
            feature_pattern = r'-\s+(.+?)(?=\n\s*-|\n\s*Most|\n\s*Against|$)'
            feature_matches = re.finditer(feature_pattern, diagnosis_text, re.DOTALL)
            for feat_match in feature_matches:
                feature = feat_match.group(1).strip()
                if feature and len(feature) > 10:
                    features.append(feature)
            
            # Extract most diagnostic patch
            patch_match = re.search(r'Most Diagnostic Patch:\s*Patch\s*(\d+)', diagnosis_text, re.IGNORECASE)
            most_diagnostic_patch = int(patch_match.group(1)) if patch_match else None
            
            # Extract against reasoning
            against_match = re.search(r'Against:\s*(.+?)(?=\n\d+\.|$)', diagnosis_text, re.DOTALL | re.IGNORECASE)
            against = against_match.group(1).strip() if against_match else ""
            
            structured['differentials'].append({
                'rank': rank,
                'diagnosis': diagnosis_name,
                'confidence': confidence,
                'supporting_features': features[:5],
                'most_diagnostic_patch': most_diagnostic_patch,
                'against_reasoning': against
            })
        
        # Extract recommended next steps
        next_steps_match = re.search(
            r'RECOMMENDED NEXT STEPS?:?\s*(.+?)(?=\n\nTEACHING|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if next_steps_match:
            structured['recommended_next_steps'] = next_steps_match.group(1).strip()
        
        # Extract teaching point
        teaching_match = re.search(
            r'TEACHING POINTS?:?\s*(.+?)$',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if teaching_match:
            structured['teaching_point'] = teaching_match.group(1).strip()
        
        return structured


# Test function
def main():
    """Test the differential diagnosis agent on RTX 3060 Ti"""
    import os
    from pathlib import Path
    
    MODEL_PATH = str(Path.home() / "pathpatch-navigator/models/medgemma-1.5-4b-it")
    PATCH_DIR = Path.home() / "pathpatch-navigator/data/patches"
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at: {MODEL_PATH}")
        return
    
    # Initialize agent with 4-bit quantization
    agent = DifferentialDiagnosisAgent(
        model_path=MODEL_PATH,
        device="cuda",
        use_4bit=True  # Critical for 8GB VRAM
    )
    
    # Load test patches
    patch_files = sorted(list(PATCH_DIR.glob("*_patch_*.png")))[:3]
    
    if len(patch_files) == 0:
        logger.error("No patches found! Please run patch extractor first.")
        return
    
    logger.info(f"Loading {len(patch_files)} test patches...")
    patches = [Image.open(p) for p in patch_files]
    
    # Clinical context
    clinical_context = {
        'age': '55',
        'gender': 'Female',
        'site': 'Breast, left upper outer quadrant',
        'specimen_type': 'Core needle biopsy',
        'stain': 'H&E'
    }
    
    # Generate differential
    result = agent.generate_differential(
        patches=patches,
        clinical_context=clinical_context,
        max_diagnoses=5
    )
    
    # Print results
    print("\n" + "="*80)
    print("DIFFERENTIAL DIAGNOSIS RESULTS")
    print("="*80)
    
    for diff in result['differentials']:
        print(f"\n{diff['rank']}. {diff['diagnosis']} - {diff['confidence']}% confidence")
        print(f"   Most Diagnostic Patch: Patch {diff['most_diagnostic_patch']}")
        print(f"   Key Features:")
        for feat in diff['supporting_features']:
            print(f"   - {feat}")
    
    print(f"\nRecommended Next Steps:\n{result['recommended_next_steps']}")
    print(f"\nTeaching Point:\n{result['teaching_point']}")
    print("="*80)


if __name__ == "__main__":
    main()