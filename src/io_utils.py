"""
I/O utilities for reading and writing CSV files with validation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


def load_csv(
    filepath: Path,
    required_columns: Optional[List[str]] = None,
    fillna: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load a CSV file with validation.
    
    Args:
        filepath: Path to CSV file
        required_columns: List of required column names
        fillna: Dictionary of default values for missing columns
    
    Returns:
        Loaded DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath, dtype=str)
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {filepath}: {missing}")
    
    if fillna:
        for col, default_val in fillna.items():
            if col in df.columns:
                df[col].fillna(default_val, inplace=True)
    
    return df


def save_csv(
    df: pd.DataFrame,
    filepath: Path,
    index: bool = False,
    float_format: Optional[str] = None,
) -> None:
    """
    Save a DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Output path
        index: Whether to save index
        float_format: Format string for floats
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index, float_format=float_format)


def create_csv_template(
    columns: List[str],
    filepath: Path,
    example_rows: Optional[List[Dict[str, str]]] = None,
) -> None:
    """
    Create a CSV template file.
    
    Args:
        columns: Column names
        filepath: Output path
        example_rows: Optional example rows
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if example_rows:
        df = pd.DataFrame(example_rows)
    else:
        df = pd.DataFrame(columns=columns)
    
    df.to_csv(filepath, index=False)


def validate_csv_schema(
    filepath: Path,
    required_columns: List[str],
    dataset_name: str = "unknown",
) -> tuple[bool, List[str]]:
    """
    Validate that a CSV has required columns.
    
    Args:
        filepath: Path to CSV
        required_columns: Required column names
        dataset_name: Name for error messages
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not filepath.exists():
        errors.append(f"File not found: {filepath}")
        return False, errors
    
    try:
        df = pd.read_csv(filepath, dtype=str)
    except Exception as e:
        errors.append(f"Failed to read CSV: {e}")
        return False, errors
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"{dataset_name}: Missing columns: {missing}")
    
    if df.empty:
        errors.append(f"{dataset_name}: CSV is empty")
    
    return len(errors) == 0, errors


def get_csv_info(filepath: Path) -> Dict[str, Any]:
    """
    Get basic info about a CSV file.
    
    Args:
        filepath: Path to CSV
    
    Returns:
        Dictionary with info
    """
    if not filepath.exists():
        return {"exists": False, "rows": 0, "columns": []}
    
    try:
        df = pd.read_csv(filepath, dtype=str, nrows=0)
        df_full = pd.read_csv(filepath, dtype=str)
        return {
            "exists": True,
            "rows": len(df_full),
            "columns": list(df.columns),
            "dtypes": df_full.dtypes.to_dict(),
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}
