"""
Batch extract patches from multiple WSI slides
Uses your existing WSIPatchExtractor
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from preprocessing.wsi_patch_extractor import WSIPatchExtractor
import json
import logging
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchPatchExtractor:
    """Batch process multiple WSI slides"""
    
    def __init__(
        self,
        input_dir,
        output_dir="../../data/processed/patches",
        metadata_dir="../../data/metadata"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractor
        self.extractor = WSIPatchExtractor(
            patch_size=256,
            magnification=20,
            tissue_threshold=0.5,  # Lowered for better extraction
            output_dir=str(self.output_dir)
        )
    
    def find_wsi_files(self):
        """Find all WSI files in input directory"""
        
        # Common WSI extensions
        extensions = ['*.svs', '*.ndpi', '*.tif', '*.tiff', '*.scn']
        
        wsi_files = []
        for ext in extensions:
            wsi_files.extend(self.input_dir.glob(f"**/{ext}"))
        
        logger.info(f"Found {len(wsi_files)} WSI files")
        return wsi_files
    
    def process_all_slides(self, num_patches=5):
        """
        Process all slides in directory
        
        Args:
            num_patches: Number of patches to extract per slide
        """
        
        logger.info("\\n" + "="*80)
        logger.info("BATCH PATCH EXTRACTION")
        logger.info("="*80)
        
        # Find all WSI files
        wsi_files = self.find_wsi_files()
        
        if not wsi_files:
            logger.error("No WSI files found!")
            return
        
        # Track results
        all_metadata = []
        successful = 0
        failed = 0
        
        # Process each slide
        for wsi_path in tqdm(wsi_files, desc="Processing slides"):
            try:
                logger.info(f"\\nProcessing: {wsi_path.name}")
                
                # Extract patches
                result = self.extractor.process_slide(
                    slide_path=str(wsi_path),
                    num_patches=num_patches,
                    save_patches=True
                )
                
                # Collect metadata
                for patch_meta in result['metadata']:
                    patch_meta['source_file'] = wsi_path.name
                    patch_meta['source_path'] = str(wsi_path)
                    all_metadata.append(patch_meta)
                
                successful += 1
                logger.info(f"✓ Extracted {result['num_patches_extracted']} patches")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {wsi_path.name}: {e}")
                failed += 1
                continue
        
        # Save combined metadata
        self._save_metadata(all_metadata)
        
        # Summary
        logger.info("\\n" + "="*80)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Successfully processed: {successful} slides")
        logger.info(f"✗ Failed: {failed} slides")
        logger.info(f"✓ Total patches extracted: {len(all_metadata)}")
        logger.info(f"✓ Patches saved to: {self.output_dir}")
        logger.info(f"✓ Metadata saved to: {self.metadata_dir}")
        logger.info("="*80)
    
    def _save_metadata(self, metadata_list):
        """Save metadata in multiple formats"""
        
        # Save as JSON
        json_path = self.metadata_dir / 'patch_metadata.json'
        with open(json_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        logger.info(f"✓ Saved JSON metadata: {json_path}")
        
        # Save as CSV (easier for analysis)
        df = pd.DataFrame(metadata_list)
        csv_path = self.metadata_dir / 'patch_metadata.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved CSV metadata: {csv_path}")
        
        # Save summary statistics
        summary = {
            'total_patches': len(metadata_list),
            'total_slides': df['slide_name'].nunique(),
            'avg_patches_per_slide': len(metadata_list) / df['slide_name'].nunique(),
            'patch_size': metadata_list[0]['patch_size'] if metadata_list else None
        }
        
        summary_path = self.metadata_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary: {summary_path}")


def main():
    """Run batch extraction"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch extract patches from WSI slides')
    parser.add_argument('--input', type=str, required=True, help='Input directory with WSI files')
    parser.add_argument('--patches', type=int, default=5, help='Number of patches per slide')
    
    args = parser.parse_args()
    
    # Run batch extraction
    extractor = BatchPatchExtractor(input_dir=args.input)
    extractor.process_all_slides(num_patches=args.patches)


if __name__ == "__main__":
    # Example usage (can also use command line)
    
    # Process TCGA slides
    print("\\n--- Processing TCGA slides ---")
    tcga_extractor = BatchPatchExtractor(
        input_dir="../../data/raw/tcga"
    )
    tcga_extractor.process_all_slides(num_patches=5)
    
    # Process DiagSet slides (if available)
    # print("\\n--- Processing DiagSet slides ---")
    # diagset_extractor = BatchPatchExtractor(
    #     input_dir="../../data/raw/diagset"
    # )
    # diagset_extractor.process_all_slides(num_patches=5)