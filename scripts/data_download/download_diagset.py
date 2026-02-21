"""
Download DiagSet dataset
Prostate cancer whole-slide images with Gleason grading
"""

import os
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagSetDownloader:
    """Download DiagSet prostate cancer dataset"""
    
    def __init__(self, output_dir="../../data/raw/diagset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # DiagSet is hosted on Zenodo
        self.dataset_url = "https://zenodo.org/record/6604028/files/DiagSet.zip"
    
    def download_dataset(self):
        """Download and extract DiagSet"""
        
        logger.info("\\n" + "="*80)
        logger.info("DIAGSET DATASET DOWNLOAD")
        logger.info("="*80)
        
        zip_path = self.output_dir / "DiagSet.zip"
        
        # Check if already downloaded
        if zip_path.exists():
            logger.info("DiagSet.zip already exists, skipping download")
        else:
            logger.info("Downloading DiagSet (~2GB, this may take a while)...")
            
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
    """Download DiagSet"""
    
    downloader = DiagSetDownloader()
    downloader.download_dataset()
    
    print("\\n✓ DiagSet download complete!")


if __name__ == "__main__":
    main()