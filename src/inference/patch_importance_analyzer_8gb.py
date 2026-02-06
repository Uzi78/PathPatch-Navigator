"""
Patch Importance Analyzer - Optimized for 8GB VRAM
Sequential processing with memory management
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from typing import List, Dict, Any
import logging
import re
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchImportanceAnalyzer:
    """
    Analyzes and ranks diagnostic importance of patches
    Optimized for 8GB VRAM
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_4bit: bool = True
    ):
        """Initialize analyzer with 4-bit quantization"""
        self.device = device
        logger.info(f"Initializing PatchImportanceAnalyzer on {device}")
        
        # Configure 4-bit quantization
        bnb_config = None
        if use_4bit and device == "cuda":
            logger.info("Using 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model
        logger.info("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("✓ Model loaded successfully!")
    
    def analyze_importance(
        self,
        patches: List[Image.Image],
        differentials: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank patches by diagnostic importance
        Processes patches SEQUENTIALLY to manage memory
        """
        logger.info(f"\nAnalyzing importance of {len(patches)} patches...")
        
        importance_results = []
        
        for idx, patch in enumerate(patches):
            # Clear cache before each patch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"Analyzing patch {idx + 1}/{len(patches)}...")
            
            prompt = self._create_importance_prompt(
                patch_number=idx + 1,
                total_patches=len(patches),
                differentials=differentials
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed = self._parse_importance(response)
            
            importance_results.append({
                'patch_id': idx + 1,
                'importance_score': parsed['score'],
                'structures_present': parsed['structures'],
                'supports_diagnoses': parsed['supports'],
                'critical_features': parsed['features'],
                'justification': parsed['justification']
            })
            
            # Clear after each iteration
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Sort by importance
        importance_results.sort(key=lambda x: x['importance_score'], reverse=True)
        
        logger.info("✓ Patch importance analysis complete")
        return importance_results
    
    def _create_importance_prompt(self, patch_number, total_patches, differentials):
        """Create prompt for importance analysis"""
        diff_list = "\n".join([
            f"- {d['diagnosis']} ({d['confidence']}%)"
            for d in differentials[:3]
        ])
        
        return f"""You are analyzing Patch {patch_number} of {total_patches} from a histopathology slide.

DIFFERENTIAL DIAGNOSES UNDER CONSIDERATION:
{diff_list}

ANALYZE THIS PATCH:
1. What histologic structures are present?
2. Which differential diagnoses does this patch support or refute?
3. Importance Score (0-10): How critical is this patch for diagnosis?
4. What would a resident miss if they skipped this patch?

Provide analysis in this format:

STRUCTURES PRESENT:
[List key structures]

SUPPORTS/REFUTES:
[Which diagnoses this patch helps with]

IMPORTANCE SCORE: [0-10]

CRITICAL FEATURES:
[What makes this patch important]

JUSTIFICATION:
[Why this score was assigned]
"""
    
    def _parse_importance(self, response):
        """Parse importance analysis response"""
        parsed = {
            'score': 5,
            'structures': [],
            'supports': [],
            'features': [],
            'justification': ''
        }
        
        score_match = re.search(r'IMPORTANCE SCORE:\s*(\d+)', response, re.IGNORECASE)
        if score_match:
            parsed['score'] = int(score_match.group(1))
        
        structures_match = re.search(
            r'STRUCTURES PRESENT:(.+?)(?=SUPPORTS|IMPORTANCE|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if structures_match:
            structures_text = structures_match.group(1)
            parsed['structures'] = [
                s.strip('- ').strip()
                for s in structures_text.split('\n')
                if s.strip() and len(s.strip()) > 5
            ]
        
        just_match = re.search(
            r'JUSTIFICATION:(.+?)$',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if just_match:
            parsed['justification'] = just_match.group(1).strip()
        
        return parsed


# Same test function as before
def main():
    """Test patch importance analyzer"""
    import os
    from pathlib import Path
    
    MODEL_PATH = str(Path.home() / "pathpatch-navigator/models/medgemma-1.5-4b-it")
    PATCH_DIR = Path.home() / "pathpatch-navigator/data/patches"
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at: {MODEL_PATH}")
        return
    
    analyzer = PatchImportanceAnalyzer(
        model_path=MODEL_PATH,
        device="cuda",
        use_4bit=True
    )
    
    patch_files = sorted(list(PATCH_DIR.glob("*_patch_*.png")))[:3]
    if len(patch_files) == 0:
        logger.error("No patches found!")
        return
    
    patches = [Image.open(p) for p in patch_files]
    
    differentials = [
        {'diagnosis': 'Invasive Ductal Carcinoma', 'confidence': 75},
        {'diagnosis': 'Invasive Lobular Carcinoma', 'confidence': 15},
        {'diagnosis': 'Fibromatosis', 'confidence': 10}
    ]
    
    results = analyzer.analyze_importance(patches, differentials)
    
    print("\n" + "="*80)
    print("PATCH IMPORTANCE RANKING")
    print("="*80)
    
    for result in results:
        print(f"\nPatch {result['patch_id']} - Importance Score: {result['importance_score']}/10")
        print(f"Structures: {', '.join(result['structures'][:3])}")
        print(f"Justification: {result['justification'][:200]}...")
    
    print("="*80)


if __name__ == "__main__":
    main()