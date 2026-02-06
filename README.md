# PathPatch Navigator

PathPatch Navigator is an AI-powered **educational assistant for pathology residents** learning whole-slide histopathology (WSI) interpretation.

The system helps users understand **how to form differential diagnoses**, **which image patches matter most**, and **how immunohistochemistry (IHC) decisions are made** — strictly as a learning aid.

## What It Does

- Analyzes **3–5 user-selected histopathology image patches**
- Generates **ranked differential diagnoses**
- Identifies **most diagnostically important patches**
- Recommends **IHC stains** to confirm or rule out diagnoses
- Produces **educational pathology report templates**


### Component Details

#### 1. **Web Interface Layer**
- **WSI Viewer**: OpenSeadragon-based whole-slide image viewer
- **Patch Selection Tool**: Interactive tool for selecting 3-5 diagnostic regions
- **Clinical Context Input**: Forms for entering relevant clinical information

#### 2. **AI Analysis Pipeline**
- **Image Processing**: OpenSlide for WSI handling and patch extraction
- **Multimodal Analysis**: MedGemma 1.5 for combined visual and text analysis
- **Feature Extraction**: Automated identification of histologic patterns

#### 3. **Agentic Workflow Engine**
- **Orchestrator**: Coordinates agent communication and data flow
- **Diagnosis Agent**: Primary diagnostic reasoning module
- **Patch Importance Agent**: Spatial analysis and feature weighting
- **IHC Recommendation Agent**: Evidence-based stain selection
- **Report Generation Agent**: Structured educational content creation

#### 4. **Educational Output Layer**
- **Interactive Reports**: Clickable explanations and references
- **Visual Guides**: Annotated patches with diagnostic markers
- **Learning Templates**: Structured educational content for residency training

### Data Flow
1. User selects patches and provides clinical context
2. System processes images and extracts features
3. Agents collaboratively analyze the case
4. Results are synthesized into educational outputs
5. User receives interactive learning materials

### Technology Integration
- **Frontend**: HTML/CSS/JavaScript with OpenSeadragon integration
- **Backend**: Python with FastAPI/Flask
- **AI Model**: MedGemma 1.5 for medical image-text understanding
- **WSI Handling**: OpenSlide for slide processing
- **Containerization**: Docker for consistent deployment

## Tech Stack

- **Python**
- **MedGemma 1.5** (multimodal medical AI)
- **OpenSlide / OpenSeadragon** (WSI handling & viewing)
- **HTML, CSS, JavaScript** (frontend)

## Intended Use

**Educational use only.**  
This project is designed to support pathology training and **must not** be used for clinical diagnosis or patient care.

## Project Status

🚧 **Prototype / In Development**

## Team Members

- Haris bin Amir  
- Muhammad Uzair  
- Muhammad Zabil  
- Talha Kausar  

## License

To be determined.
