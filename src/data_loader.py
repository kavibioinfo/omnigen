"""
OmniGen - Data Loading Module
Loads and preprocesses DRKG data for drug repurposing.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


class DRKGLoader:
    """Handles loading and basic exploration of DRKG dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.df: Optional[pd.DataFrame] = None
        
    def load(self, filepath: str = "drkg.tsv") -> pd.DataFrame:
        """Load DRKG triplets from TSV file."""
        path = self.data_dir / filepath
        
        if not path.exists():
            raise FileNotFoundError(
                f"DRKG file not found at {path}\n"
                f"Please download from: https://github.com/gnn4dr/DRKG"
            )
        
        self.df = pd.read_csv(
            path, 
            sep="\t", 
            header=None,
            names=["head", "relation", "tail"]
        )
        
        print(f"Loaded {len(self.df):,} triplets")
        print(f"Unique entities: {self.df['head'].nunique() + self.df['tail'].nunique():,}")
        print(f"Unique relations: {self.df['relation'].nunique()}")
        
        return self.df
    
    def get_entity_types(self) -> Dict[str, int]:
        """Count occurrences of each entity type."""
        if self.df is None:
            raise ValueError("Load data first with .load()")
        
        types = self.df["head"].str.split("::").str[0]
        return types.value_counts().to_dict()
    
    def get_relation_types(self) -> pd.Series:
        """Get frequency of each relation type."""
        if self.df is None:
            raise ValueError("Load data first with .load()")
        return self.df["relation"].value_counts()
    
    def sample_for_testing(self, n: int = 10000) -> pd.DataFrame:
        """Get a small sample for quick testing."""
        if self.df is None:
            raise ValueError("Load data first with .load()")
        return self.df.sample(n=n, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    loader = DRKGLoader()
    print("DRKGLoader ready! Call .load() after downloading DRKG.")