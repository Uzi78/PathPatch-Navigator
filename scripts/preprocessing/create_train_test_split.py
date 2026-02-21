"""
Create train/validation/test splits from extracted patches
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Create train/val/test splits"""
    
    def __init__(
        self,
        metadata_path="../../data/metadata/patch_metadata.csv",
        output_dir="../../data/splits"
    ):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_splits(
        self,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    ):
        """
        Create stratified splits
        
        Args:
            train_ratio: Proportion for training (0.7 = 70%)
            val_ratio: Proportion for validation (0.15 = 15%)
            test_ratio: Proportion for testing (0.15 = 15%)
        """
        
        logger.info("\\n" + "="*80)
        logger.info("CREATING TRAIN/VAL/TEST SPLITS")
        logger.info("="*80)
        
        # Load metadata
        logger.info(f"Loading metadata from: {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        
        logger.info(f"Total patches: {len(df)}")
        logger.info(f"Total slides: {df['slide_name'].nunique()}")
        
        # Split by slide (not by patch) to avoid data leakage
        unique_slides = df['slide_name'].unique()
        
        # First split: train + (val+test)
        train_slides, temp_slides = train_test_split(
            unique_slides,
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Second split: val + test
        val_slides, test_slides = train_test_split(
            temp_slides,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Create splits
        train_df = df[df['slide_name'].isin(train_slides)]
        val_df = df[df['slide_name'].isin(val_slides)]
        test_df = df[df['slide_name'].isin(test_slides)]
        
        # Save splits
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'validation.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        
        # Save split info
        split_info = {
            'train': {
                'num_patches': len(train_df),
                'num_slides': len(train_slides),
                'ratio': train_ratio
            },
            'validation': {
                'num_patches': len(val_df),
                'num_slides': len(val_slides),
                'ratio': val_ratio
            },
            'test': {
                'num_patches': len(test_df),
                'num_slides': len(test_slides),
                'ratio': test_ratio
            }
        }
        
        with open(self.output_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        # Summary
        logger.info("\\n" + "="*80)
        logger.info("SPLITS CREATED")
        logger.info("="*80)
        logger.info(f"Train: {len(train_df)} patches from {len(train_slides)} slides")
        logger.info(f"Val:   {len(val_df)} patches from {len(val_slides)} slides")
        logger.info(f"Test:  {len(test_df)} patches from {len(test_slides)} slides")
        logger.info(f"\\nFiles saved to: {self.output_dir}")
        logger.info("="*80)


def main():
    """Create splits"""
    
    splitter = DatasetSplitter()
    splitter.create_splits(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )


if __name__ == "__main__":
    main()