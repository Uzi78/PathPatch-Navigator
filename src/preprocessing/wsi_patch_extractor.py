"""
WSI Patch Extraction Module
Extracts diagnostically relevant patches from whole-slide images
"""

import openslide
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
from typing import List, Tuple, Dict
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WSIPatchExtractor:
    """
    Extracts patches from whole-slide images for PathPatch Navigator
    """
    
    def __init__(
        self,
        patch_size: int = 256,
        magnification: int = 20,
        tissue_threshold: float = 0.7,
        output_dir: str = "./data/patches"
    ):
        """
        Initialize patch extractor
        
        Args:
            patch_size: Size of extracted patches (256x256 recommended)
            magnification: Target magnification level (10x or 20x)
            tissue_threshold: Minimum tissue percentage in patch (0-1)
            output_dir: Directory to save extracted patches
        """
        self.patch_size = patch_size
        self.magnification = magnification
        self.tissue_threshold = tissue_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized WSIPatchExtractor:")
        logger.info(f"  Patch size: {patch_size}x{patch_size}")
        logger.info(f"  Magnification: {magnification}x")
        logger.info(f"  Tissue threshold: {tissue_threshold}")
    
    def load_slide(self, slide_path: str) -> openslide.OpenSlide:
        """Load a whole-slide image"""
        logger.info(f"Loading slide: {slide_path}")
        slide = openslide.OpenSlide(slide_path)
        
        # Log slide properties
        logger.info(f"  Dimensions: {slide.dimensions}")
        logger.info(f"  Level count: {slide.level_count}")
        logger.info(f"  Level dimensions: {slide.level_dimensions}")
        logger.info(f"  Objective power: {slide.properties.get('openslide.objective-power', 'Unknown')}")
        
        return slide
    
    def get_tissue_mask(self, slide: openslide.OpenSlide, level: int = None) -> Tuple[np.ndarray, int]:
        """
        Generate tissue mask to identify tissue regions vs. background
        
        Args:
            slide: OpenSlide object
            level: Pyramid level for mask generation (higher = faster). If None, uses last available level.
        
        Returns:
            Tuple of (Binary mask where 1 = tissue, 0 = background, level used)
        """
        logger.info("Generating tissue mask...")
        
        # Use the last available level if not specified
        if level is None:
            level = slide.level_count - 1
        
        # Ensure level is within bounds
        level = min(level, slide.level_count - 1)
        logger.info(f"  Using pyramid level {level} for tissue mask")
        
        # Read thumbnail at low resolution
        thumbnail = slide.read_region(
            (0, 0),
            level,
            slide.level_dimensions[level]
        )
        thumbnail = np.array(thumbnail.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        
        # Otsu thresholding to separate tissue from background
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert to binary
        mask = (mask > 0).astype(np.uint8)
        
        logger.info(f"  Tissue coverage: {np.mean(mask) * 100:.2f}%")
        
        return mask, level
    
    def calculate_patch_coordinates(
        self,
        slide: openslide.OpenSlide,
        tissue_mask: np.ndarray,
        mask_level: int,
        num_patches: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Calculate coordinates for extracting patches
        
        Args:
            slide: OpenSlide object
            tissue_mask: Binary tissue mask
            mask_level: The pyramid level used for the tissue mask
            num_patches: Number of patches to extract
        
        Returns:
            List of (x, y) coordinates at level 0
        """
        logger.info(f"Calculating coordinates for {num_patches} patches...")
        
        # Get downsample factor for the mask level
        downsample = slide.level_downsamples[mask_level]
        
        # Find tissue regions
        tissue_coords = np.argwhere(tissue_mask > 0)
        
        if len(tissue_coords) == 0:
            raise ValueError("No tissue found in slide!")
        
        # Calculate patch spacing for even distribution
        patch_spacing = int(len(tissue_coords) / (num_patches + 1))
        
        # Select evenly spaced patches
        selected_coords = []
        for i in range(num_patches):
            idx = (i + 1) * patch_spacing
            if idx < len(tissue_coords):
                y_mask, x_mask = tissue_coords[idx]
                
                # Convert to level 0 coordinates
                x_level0 = int(x_mask * downsample)
                y_level0 = int(y_mask * downsample)
                
                # Center the patch
                x_level0 -= self.patch_size // 2
                y_level0 -= self.patch_size // 2
                
                # Ensure within bounds
                x_level0 = max(0, min(x_level0, slide.dimensions[0] - self.patch_size))
                y_level0 = max(0, min(y_level0, slide.dimensions[1] - self.patch_size))
                
                selected_coords.append((x_level0, y_level0))
        
        logger.info(f"  Selected {len(selected_coords)} patch locations")
        return selected_coords
    
    def extract_patch(
        self,
        slide: openslide.OpenSlide,
        x: int,
        y: int
    ) -> Image.Image:
        """
        Extract a single patch from slide
        
        Args:
            slide: OpenSlide object
            x: X coordinate at level 0
            y: Y coordinate at level 0
        
        Returns:
            PIL Image of patch
        """
        patch = slide.read_region(
            (x, y),
            0,  # Level 0 (highest resolution)
            (self.patch_size, self.patch_size)
        )
        
        # Convert RGBA to RGB
        patch = patch.convert('RGB')
        
        return patch
    
    def is_patch_valid(self, patch: Image.Image) -> bool:
        """
        Check if patch contains sufficient tissue
        
        Args:
            patch: PIL Image
        
        Returns:
            True if patch is valid
        """
        # Convert to numpy array
        patch_np = np.array(patch)
        
        # Convert to grayscale
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate tissue percentage
        tissue_pct = np.mean(binary > 0)
        
        return tissue_pct >= self.tissue_threshold
    
    def process_slide(
        self,
        slide_path: str,
        num_patches: int = 5,
        save_patches: bool = True
    ) -> Dict:
        """
        Process entire WSI and extract patches
        
        Args:
            slide_path: Path to .svs file
            num_patches: Number of patches to extract
            save_patches: Whether to save patches to disk
        
        Returns:
            Dictionary containing patches and metadata
        """
        slide_name = Path(slide_path).stem
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing slide: {slide_name}")
        logger.info(f"{'='*80}")
        
        # Load slide
        slide = self.load_slide(slide_path)
        
        # Generate tissue mask
        tissue_mask, mask_level = self.get_tissue_mask(slide)
        
        # Calculate patch coordinates
        coords = self.calculate_patch_coordinates(
            slide, tissue_mask, mask_level, num_patches
        )
        
        # Extract patches
        patches = []
        patch_metadata = []
        
        for idx, (x, y) in enumerate(coords):
            logger.info(f"Extracting patch {idx + 1}/{len(coords)} at ({x}, {y})")
            
            # Extract patch
            patch = self.extract_patch(slide, x, y)
            
            # Validate patch
            if not self.is_patch_valid(patch):
                logger.warning(f"  Patch {idx + 1} failed tissue threshold, skipping")
                continue
            
            # Save patch if requested
            if save_patches:
                patch_filename = f"{slide_name}_patch_{idx:03d}.png"
                patch_path = self.output_dir / patch_filename
                patch.save(patch_path)
                logger.info(f"  Saved to: {patch_path}")
            
            # Store patch and metadata
            patches.append(patch)
            patch_metadata.append({
                'patch_id': idx,
                'slide_name': slide_name,
                'coordinates': {'x': x, 'y': y},
                'patch_size': self.patch_size,
                'magnification': self.magnification
            })
        
        # Close slide
        slide.close()
        
        # Create result dictionary
        result = {
            'slide_name': slide_name,
            'slide_path': slide_path,
            'num_patches_extracted': len(patches),
            'patches': patches,
            'metadata': patch_metadata
        }
        
        # Save metadata
        if save_patches:
            metadata_path = self.output_dir / f"{slide_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                # Convert to JSON-serializable format
                json_metadata = {
                    'slide_name': slide_name,
                    'num_patches': len(patches),
                    'patches': patch_metadata
                }
                json.dump(json_metadata, f, indent=2)
            logger.info(f"Saved metadata to: {metadata_path}")
        
        logger.info(f"\n✓ Successfully extracted {len(patches)} patches from {slide_name}")
        
        return result


# Test function
def main():
    """Test the patch extractor"""
    
    # Initialize extractor
    extractor = WSIPatchExtractor(
        patch_size=256,
        magnification=20,
        tissue_threshold=0.7,
        output_dir="./data/patches"
    )
    
    # Process a test slide
    slide_path = "./data/wsi_slides/CMU-1.svs"
    
    if not os.path.exists(slide_path):
        logger.error(f"Slide not found: {slide_path}")
        logger.error("Please download a test slide first!")
        return
    
    # Extract patches
    result = extractor.process_slide(
        slide_path=slide_path,
        num_patches=5,
        save_patches=True
    )
    
    logger.info("\n" + "="*80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Patches extracted: {result['num_patches_extracted']}")
    logger.info(f"Output directory: {extractor.output_dir}")


if __name__ == "__main__":
    main()