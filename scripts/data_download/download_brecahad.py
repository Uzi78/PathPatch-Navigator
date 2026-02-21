"""
Download BreCaHAD dataset
Breast Cancer Histopathological Annotation and Diagnosis
"""

import os
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BreCaHADDownloader:
    """Download BreCaHAD dataset"""
    
    def __init__(self, output_dir="../../data/raw/brecahad"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # BreCaHAD dataset URL (from figshare)
        self.dataset_url = "https://figshare.com/ndownloader/files/21439597"
    
    def download_dataset(self):
        """Download and extract BreCaHAD"""
        
        logger.info("\\n" + "="*80)
        logger.info("BRECAHAD DATASET DOWNLOAD")
        logger.info("="*80)
        
        zip_path = self.output_dir / "BreCaHAD.zip"
        
        # Check if already downloaded
        if zip_path.exists():
            logger.info("BreCaHAD.zip already exists, skipping download")
        else:
            logger.info("Downloading BreCaHAD (~500MB)...")
            
            response = requests.get(self.dataset_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info("✓ Download complete")
        
        # Extract
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)
        
        logger.info("✓ Extraction complete")
        logger.info(f"✓ Dataset saved to: {self.output_dir}")
        logger.info("="*80)


def main():
    """Download BreCaHAD"""
    
    downloader = BreCaHADDownloader()
    downloader.download_dataset()
    
    print("\\n✓ BreCaHAD download complete!")


if __name__ == "__main__":
    main()