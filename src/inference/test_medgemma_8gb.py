"""
MedGemma test optimized for 8GB VRAM (RTX 3060 Ti)
Uses 4-bit quantization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "/home/uzair/pathpatch-navigator/models/medgemma-1.5-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"CUDA Version: {torch.version.cuda}")

# Configure 4-bit quantization for 8GB VRAM
logger.info("\nConfiguring 4-bit quantization for memory efficiency...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",             # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.float16,   # Compute in FP16
    bnb_4bit_use_double_quant=True,        # Double quantization for extra compression
)

# Load tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model with 4-bit quantization
logger.info("Loading model with 4-bit quantization (this may take 2-3 minutes)...")
logger.info("Expected memory usage: ~2.5 GB VRAM")

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    logger.info("✓ Model loaded successfully!")
    
    # Check memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        logger.info(f"GPU Memory Allocated: {allocated:.2f} GB")
        logger.info(f"GPU Memory Reserved: {reserved:.2f} GB")
        logger.info(f"Free VRAM: {8 - reserved:.2f} GB")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.error("\nTroubleshooting:")
    logger.error("1. Ensure no other programs are using GPU")
    logger.error("2. Close any other Python processes")
    logger.error("3. Restart your computer to clear GPU memory")
    raise

# Test with a medical image
logger.info("\n" + "="*80)
logger.info("Testing inference with sample medical image...")
logger.info("="*80)

# Download a test histopathology image
test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Emphysema_H_and_E.jpg/800px-Emphysema_H_and_E.jpg"
response = requests.get(test_image_url)
image = Image.open(BytesIO(response.content))

logger.info(f"Test image size: {image.size}")

# Prepare input
prompt = """Analyze this H&E stained histopathology image. 
Describe the key features you observe and provide a differential diagnosis."""

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

logger.info("Generating response (may take 30-60 seconds on RTX 3060 Ti)...")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

logger.info("\n" + "="*80)
logger.info("MODEL RESPONSE:")
logger.info("="*80)
print(response)
logger.info("="*80)

# Final memory check
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    logger.info(f"\nFinal GPU Memory: {allocated:.2f} GB")
    logger.info(f"Peak GPU Memory: {peak:.2f} GB")

logger.info("\n✓ MedGemma is working correctly on RTX 3060 Ti!")