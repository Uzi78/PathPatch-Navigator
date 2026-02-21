"""
Download TCGA-BRCA whole-slide images
Breast cancer dataset from The Cancer Genome Atlas
"""

import os
import requests
from pathlib import Path
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TCGADownloader:
    """Download TCGA-BRCA dataset"""
    
    def __init__(self, output_dir="../../data/raw/tcga"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GDC API endpoint
        self.base_url = "https://api.gdc.cancer.gov"
        
    def get_tcga_slide_ids(self, num_slides=10):
        """
        Get slide IDs from GDC Data Portal
        
        Args:
            num_slides: Number of slides to download (start small for testing)
        """
        logger.info(f"Fetching {num_slides} TCGA-BRCA slide IDs...")
        
        # Query GDC for TCGA-BRCA slides
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": ["TCGA-BRCA"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_type",
                        "value": ["Slide Image"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.experimental_strategy",
                        "value": ["Diagnostic Slide"]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,cases.diagnoses.primary_diagnosis",
            "format": "JSON",
            "size": num_slides
        }
        
        response = requests.post(
            f"{self.base_url}/files",
            headers={"Content-Type": "application/json"},
            json=params
        )
        
        if response.status_code == 200:
            data = response.json()
            slides = data['data']['hits']
            logger.info(f"✓ Found {len(slides)} slides")
            return slides
        else:
            logger.error(f"Failed to fetch slides: {response.status_code}")
            return []
    
    def download_slide(self, file_id, file_name):
        """Download a single slide"""
        
        output_path = self.output_dir / file_name
        
        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"Skipping {file_name} (already exists)")
            return True
        
        logger.info(f"Downloading {file_name}...")
        
        url = f"{self.base_url}/data/{file_id}"
        
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"✓ Downloaded {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_dataset(self, num_slides=10):
        """Download multiple slides"""
        
        logger.info("\\n" + "="*80)
        logger.info("TCGA-BRCA DATASET DOWNLOAD")
        logger.info("="*80)
        
        # Get slide IDs
        slides = self.get_tcga_slide_ids(num_slides)
        
        if not slides:
            logger.error("No slides found!")
            return
        
        # Create metadata
        metadata = []
        
        # Download each slide
        for slide in slides:
            file_id = slide['file_id']
            file_name = slide['file_name']
            file_size = slide['file_size']
            
            # Get diagnosis if available
            diagnosis = "Unknown"
            if 'cases' in slide and len(slide['cases']) > 0:
                if 'diagnoses' in slide['cases'][0] and len(slide['cases'][0]['diagnoses']) > 0:
                    diagnosis = slide['cases'][0]['diagnoses'][0].get('primary_diagnosis', 'Unknown')
            
            success = self.download_slide(file_id, file_name)
            
            if success:
                metadata.append({
                    'file_id': file_id,
                    'file_name': file_name,
                    'file_size_mb': file_size / (1024**2),
                    'diagnosis': diagnosis,
                    'source': 'TCGA-BRCA'
                })
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\\n✓ Downloaded {len(metadata)} slides")
        logger.info(f"✓ Metadata saved to: {metadata_path}")
        logger.info("="*80)


def main():
    """Download TCGA dataset"""
    
    # NOTE: Start with small number for testing
    # Full dataset has 1000+ slides (several TB)
    NUM_SLIDES = 5  # Start with 5 for testing
    
    downloader = TCGADownloader()
    downloader.download_dataset(num_slides=NUM_SLIDES)
    
    print("\\n✓ TCGA download complete!")
    print(f"Files saved to: {downloader.output_dir}")


if __name__ == "__main__":
    main()