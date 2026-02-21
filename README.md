# PathPatch Navigator

An AI-powered educational assistant for pathology residents learning whole-slide histopathology (WSI) interpretation with multimodal reasoning and evidence-based IHC recommendations.

**Disclaimer:** Educational use only. Must not be used for clinical diagnosis or patient care.

## Overview

PathPatch Navigator helps pathology trainees understand differential diagnoses, identify diagnostically important image regions, select appropriate IHC stains, and generate structured educational reports.

## Key Features

- Multi-patch analysis (3-5 patches simultaneously)
- Differential diagnosis generation with confidence scores
- Patch importance ranking
- IHC recommendation engine with evidence-based stain selection
- Cost & availability analysis with institutional pricing
- CAP-aligned report generation
- 4-bit quantized inference (optimized for 8GB VRAM)
- Memory-efficient sequential processing
- MedGemma 1.5 multimodal AI
- OpenSlide WSI support
- Structured JSON output
- FastAPI REST API

## Architecture

5-agent orchestration pipeline:
1. Diagnosis Agent - Generate ranked differential diagnoses from patches
2. Patch Importance Agent - Rank submitted patches by diagnostic value  
3. IHC Recommendation Agent - Recommend immunohistochemistry stains
4. Cost Calculator Agent - Calculate testing costs and institutional availability
5. Report Generator Agent - Create CAP-aligned pathology report templates

Agents coordinate through `AgentOrchestrator` and output structured JSON via `PathPatchJSONFormatter`.

## Installation

```bash
cd pathpatch-navigator
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Download MedGemma model:
```bash
huggingface-cli download googlelabs/medgemma-1.5-4b-it --local-dir models/medgemma-1.5-4b-it
```

Run tests:
```bash
python scripts/run_smoke_local.py
python tests/test_full_pipeline.py
python tests/test_on_tcga.py
```

## API Usage

Start server:
```bash
python src/api/main.py
```

POST to `http://localhost:8000/api/v1/analyze` with multipart form data (3-5 PNG patches + clinical context fields).

Response includes: differential diagnoses, patch importance rankings, IHC recommendations, cost analysis, and CAP-aligned report template.

## Data Structure

Input data: PNG/JPG patches (256x256 at 20x magnification)

Directory structure:
```
data/
├── metadata/: Patch extraction metadata
├── patches/: Extracted patches per slide
├── processed/: Post-processed patches
├── raw/: BreCaHAD, DIAGSET, TCGA datasets
├── splits/: Train/val/test splits
└── wsi_slides/: .svs whole-slide images
```

Output: `./output/<case_id>_<timestamp>_results.json`

## Scripts

Data preparation:
- `scripts/preprocessing/batch_extract_patches.py` - Extract patches from WSI
- `scripts/preprocessing/create_train_test_split.py` - Generate splits
- `scripts/data_download/download_tcga.py` - Download TCGA slides
- `scripts/data_download/download_diagset.py` - Download DIAGSET
- `scripts/data_download/download_brecahad.py` - Download BreCaHAD

Testing:
- `scripts/run_smoke_local.py` - Lightweight test without large models
- `tests/test_full_pipeline.py` - End-to-end pipeline with MedGemma
- `tests/test_integration_orchestrator.py` - Agent orchestrator tests
- `tests/test_on_tcga.py` - Multi-slide TCGA evaluation

## Knowledge Bases

IHC Database (src/knowledge_base/ihc_database.py): 20+ stains for breast, prostate, epithelial, squamous, neuroendocrine, lymphoid pathology

Cost Database (src/knowledge_base/cost_database.py): Institutional pricing (default: ER/PR $60, HER2 $80, Ki-67 $75, resident approval limit $300)

CAP Guidelines (src/knowledge_base/cap_guidelines.py): Diagnostic criteria, Nottingham grading, Gleason scoring, synoptic elements

Supported diagnoses: Invasive Ductal Carcinoma, Invasive Lobular Carcinoma, Prostate Adenocarcinoma

## Inference Modules

Differential Diagnosis Agent (src/inference/differential_diagnosis_agent_8gb.py): MedGemma 1.5 with 4-bit quantization for 8GB VRAM

Patch Importance Analyzer (src/inference/patch_importance_analyzer_8gb.py): Sequential patch ranking with memory management

## Requirements & Tech Stack

Dependencies:
```
bitsandbytes, fastapi, huggingface_hub, numpy, opencv-python
openslide-python, Pillow, pydantic, python-multipart, requests
safetensors, sentencepiece, tokenizers, torch, transformers
uvicorn, loguru
```

Tech Stack:
- Backend: FastAPI + Uvicorn
- AI/ML: MedGemma 1.5 (4-bit quantized)
- Image Processing: OpenSlide, OpenCV, Pillow
- Data: Pydantic, JSON
- Logging: Loguru



## Project Status

Complete:
- Core agentic workflow (5 agents)
- MedGemma inference optimization
- IHC recommendations engine
- Cost calculations with approvals
- CAP-aligned report generation
- REST API with multipart support
- 4-bit quantization for 8GB VRAM
- Comprehensive testing suite

## Team

- Haris bin Amir
- Muhammad Uzair
- Muhammad Zabil
- Talha Kausar

## License

To be determined

## Resources

- MedGemma: https://huggingface.co/googlelabs/medgemma-1.5-4b-it
- OpenSlide: http://openslide.org/
- FastAPI: https://fastapi.tiangolo.com/
- CAP Guidelines: https://www.cap.org/guides-and-quality-standards/cap-synoptic-checklists

Last Updated: February 21, 2026
Version: 1.0.0
