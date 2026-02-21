"""
PathPatch Navigator REST API - Optimized for 8GB VRAM
Uses 4-bit quantization and memory management
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from PIL import Image
import io
import logging
from pathlib import Path
import sys
import torch
import gc
import os
import asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.differential_diagnosis_agent_8gb import DifferentialDiagnosisAgent
from inference.patch_importance_analyzer_8gb import PatchImportanceAnalyzer
from utils.json_formatter import PathPatchJSONFormatter
from agents.diagnosis_agent import DiagnosisAgent
from agents.patch_importance_agent import PatchImportanceAgent as PatchAgentWrapper
from agents.ihc_recommendation_agent import IHCRecommendationAgent
from agents.cost_calculator_agent import CostCalculatorAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.orchestrator import AgentOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PathPatch Navigator API (8GB VRAM Optimized)",
    description="AI-powered histopathology analysis optimized for RTX 3060 Ti",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = str(Path.home() / "pathpatch-navigator/models/medgemma-1.5-4b-it")
OUTPUT_DIR = "./output"

diagnosis_inference = None
importance_inference = None
json_formatter = None
orchestrator = None

# Concurrency limiter for GPU-bound analysis
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "1"))
ANALYSIS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)


class ClinicalContext(BaseModel):
    age: Optional[str] = "Unknown"
    gender: Optional[str] = "Unknown"
    site: str
    specimen_type: str = "Biopsy"
    stain: str = "H&E"
    additional_info: Optional[str] = None


class AnalysisResponse(BaseModel):
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.on_event("startup")
async def load_models():
    """Load models with 4-bit quantization"""
    global diagnosis_inference, importance_inference, json_formatter, orchestrator
    
    logger.info("="*80)
    logger.info("PATHPATCH NAVIGATOR API - STARTUP (8GB VRAM MODE)")
    logger.info("="*80)
    
    try:
        logger.info("Loading DifferentialDiagnosisAgent with 4-bit quantization...")
        diagnosis_inference = DifferentialDiagnosisAgent(
            model_path=MODEL_PATH,
            device="cuda",
            use_4bit=True  # Critical for 8GB VRAM
        )
        
        logger.info("Loading PatchImportanceAnalyzer with 4-bit quantization...")
        importance_inference = PatchImportanceAnalyzer(
            model_path=MODEL_PATH,
            device="cuda",
            use_4bit=True
        )
        
        logger.info("Initializing JSONFormatter...")
        json_formatter = PathPatchJSONFormatter(output_dir=OUTPUT_DIR)

        logger.info("Configuring agentic workflow components...")
        diagnosis_agent = DiagnosisAgent(diagnosis_inference)
        patch_agent = PatchAgentWrapper(importance_inference)
        ihc_agent = IHCRecommendationAgent()
        cost_agent = CostCalculatorAgent()
        report_agent = ReportGeneratorAgent()

        orchestrator = AgentOrchestrator(
            diagnosis_agent=diagnosis_agent,
            patch_agent=patch_agent,
            ihc_agent=ihc_agent,
            cost_agent=cost_agent,
            report_agent=report_agent,
            json_formatter=json_formatter,
        )
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
        
        logger.info("="*80)
        logger.info("✓ All models loaded successfully!")
        logger.info("✓ API ready to accept requests")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/")
async def root():
    return {
        "message": "PathPatch Navigator API (8GB VRAM Optimized)",
        "version": "1.0.0",
        "gpu": "RTX 3060 Ti",
        "optimization": "4-bit quantization",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    gpu_status = {}
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_status = {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total, 2),
            "vram_allocated_gb": round(allocated, 2),
            "vram_free_gb": round(total - allocated, 2)
        }
    
    return {
        "status": "healthy",
        "models_loaded": all([
            diagnosis_inference is not None,
            importance_inference is not None,
            orchestrator is not None,
        ]),
        "gpu_status": gpu_status,
        "agents": [
            "DiagnosisAgent",
            "PatchImportanceAgent",
            "IHCRecommendationAgent",
            "CostCalculatorAgent",
            "ReportGeneratorAgent",
        ],
    }


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_patches(
    patches: List[UploadFile] = File(..., description="3-5 histopathology patches"),
    slide_name: str = Form(...),
    age: str = Form(default="Unknown"),
    gender: str = Form(default="Unknown"),
    site: str = Form(...),
    specimen_type: str = Form(default="Biopsy"),
    stain: str = Form(default="H&E"),
    max_diagnoses: int = Form(default=5)
):
    """Analyze patches with memory management for 8GB VRAM"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"NEW ANALYSIS REQUEST: {slide_name}")
    logger.info(f"{'='*80}")
    
    try:
        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Validate
        if len(patches) < 3 or len(patches) > 5:
            raise HTTPException(
                status_code=400,
                detail=f"Please provide 3-5 patches (received {len(patches)})"
            )
        
        # Load images
        logger.info(f"Loading {len(patches)} patches...")
        patch_images = []
        patches_metadata = []
        
        for idx, patch_file in enumerate(patches):
            contents = await patch_file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            patch_images.append(image)
            
            patches_metadata.append({
                'patch_id': idx + 1,
                'filename': patch_file.filename,
                'patch_size': image.size[0]
            })
            
            logger.info(f"  Patch {idx + 1}: {patch_file.filename} ({image.size})")
        
        clinical_context = {
            'age': age,
            'gender': gender,
            'site': site,
            'specimen_type': specimen_type,
            'stain': stain
        }
        
        if orchestrator is None:
            raise HTTPException(status_code=503, detail="Workflow orchestrator not ready")

        # Rate-limit concurrent analyses to avoid GPU OOM
        await ANALYSIS_SEMAPHORE.acquire()
        try:
            workflow_payload = await orchestrator.execute_workflow(
                {
                    "patches": patch_images,
                    "patch_metadata": patches_metadata,
                    "clinical_context": clinical_context,
                    "slide_name": slide_name,
                    "max_diagnoses": max_diagnoses,
                }
            )
        finally:
            ANALYSIS_SEMAPHORE.release()

        status = workflow_payload.get("status", "error")
        if status == "error":
            raise HTTPException(status_code=500, detail=workflow_payload.get("error", "Unknown error"))

        top_diag = workflow_payload.get("diagnosis", {}).get("differentials", [])
        if top_diag:
            top_entry = top_diag[0]
            logger.info(
                f"Top diagnosis: {top_entry['diagnosis']} ({top_entry['confidence']}%)"
            )

        return AnalysisResponse(
            status="success",
            message=f"Successfully analyzed {len(patches)} patches",
            results=workflow_payload
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting PathPatch Navigator API (8GB VRAM Optimized)...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )