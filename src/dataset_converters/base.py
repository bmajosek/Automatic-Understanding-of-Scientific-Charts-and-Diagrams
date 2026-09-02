"""
Base dataset converter.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd


class BaseConverter(ABC):
    """Base class for dataset converters."""
    
    def __init__(self, raw_data_dir: Path, output_dir: Path):
        """
        Initialize converter.
        
        Args:
            raw_data_dir: Directory containing raw dataset files
            output_dir: Directory where to save converted files
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.warnings = []
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Dataset name."""
        pass
    
    @abstractmethod
    def check_raw_data(self) -> bool:
        """Check if raw data exists."""
        pass
    
    @abstractmethod
    def convert(self) -> Dict[str, pd.DataFrame]:
        """
        Convert raw dataset to standard format.
        
        Returns:
            Dictionary with keys like 'charts', 'tables', 'questions', etc.
        """
        pass
    
    def save_output(self, output_dfs: Dict[str, pd.DataFrame]) -> None:
        """Save converted datasets."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in output_dfs.items():
            if df is not None and not df.empty:
                filepath = self.output_dir / f"{name}.csv"
                df.to_csv(filepath, index=False)
    
    def save_warnings(self, warnings_file: Path) -> None:
        """Save conversion warnings."""
        if self.warnings:
            warnings_file.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(self.warnings)
            df.to_csv(warnings_file, index=False)
    
    def add_warning(
        self,
        dataset_name: str,
        original_id: str,
        reason: str,
        raw_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a conversion warning."""
        self.warnings.append({
            'dataset_name': dataset_name,
            'original_id': original_id,
            'reason': reason,
            'raw_fields_available': str(raw_fields or {}),
        })
