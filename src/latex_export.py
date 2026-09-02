"""
LaTeX table export.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class LaTeXExporter:
    """Generate LaTeX tables for thesis."""
    
    @staticmethod
    def escape_latex(text: str) -> str:
        """Escape special LaTeX characters."""
        if not isinstance(text, str):
            return str(text)
        
        replacements = [
            ('\\', r'\textbackslash{}'),
            ('&', r'\&'),
            ('%', r'\%'),
            ('$', r'\$'),
            ('#', r'\#'),
            ('_', r'\_'),
            ('{', r'\{'),
            ('}', r'\}'),
            ('~', r'\textasciitilde{}'),
            ('^', r'\textasciicircum{}'),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def format_float(value: Any, decimals: int = 3) -> str:
        """Format float for LaTeX."""
        if pd.isna(value) or np.isnan(value):
            return '--'
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def dataframe_to_latex(
        df: pd.DataFrame,
        caption: str,
        label: str,
        filepath: Path,
        float_format: str = '.3f',
        escape: bool = True,
    ) -> None:
        """
        Convert DataFrame to LaTeX table.
        
        Args:
            df: DataFrame to export
            caption: Table caption
            label: Table label (for referencing)
            filepath: Output file path
            float_format: Format for floats
            escape: Whether to escape special characters
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Escape column names
        if escape:
            df.columns = [LaTeXExporter.escape_latex(str(col)) for col in df.columns]
            df = df.map(
                lambda x: LaTeXExporter.escape_latex(str(x)) if not pd.isna(x) else '--'
            )
        
        # Generate LaTeX
        latex_str = df.to_latex(index=False, float_format=float_format)
        
        # Wrap in table environment
        table_str = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{tab:{label}}}
\\small
{latex_str}
\\end{{table}}
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(table_str)
    
    @staticmethod
    def format_metrics_table(
        results: List[Dict],
        metric_keys: List[str],
        model_key: str = 'model_name',
        filepath: Optional[Path] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        float_decimals: int = 3,
    ) -> pd.DataFrame:
        """
        Format a metrics table from results list.
        
        Args:
            results: List of result dictionaries
            metric_keys: Which metrics to include
            model_key: Key for model names
            filepath: If provided, save to LaTeX
            caption: Table caption
            label: Table label
            float_decimals: Number of decimals for floats
        
        Returns:
            Formatted DataFrame
        """
        rows = []
        for result in results:
            row = {'Model': result.get(model_key, 'Unknown')}
            for key in metric_keys:
                value = result.get(key, np.nan)
                if isinstance(value, float):
                    row[key] = LaTeXExporter.format_float(value, float_decimals)
                else:
                    row[key] = str(value)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if filepath and caption and label:
            LaTeXExporter.dataframe_to_latex(df, caption, label, filepath)
        
        return df
    
    @staticmethod
    def generate_qa_table(
        qa_results: List[Dict],
        filepath: Path,
    ) -> None:
        """Generate QA results table."""
        rows = []
        for result in qa_results:
            rows.append({
                'Model': result['model_name'],
                'Evaluated': str(result.get('evaluated_questions', 0)),
                'Correct': str(result.get('correct_answers', 0)),
                'Accuracy': LaTeXExporter.format_float(result.get('accuracy')),
                'Numerical MAE': LaTeXExporter.format_float(result.get('numerical_mae')),
                'Numerical MRE': LaTeXExporter.format_float(result.get('numerical_mre')),
            })
        
        df = pd.DataFrame(rows)
        
        LaTeXExporter.dataframe_to_latex(
            df,
            caption='Chart Question Answering Results',
            label='qa_overall',
            filepath=filepath,
        )
    
    @staticmethod
    def generate_table_extraction_table(
        table_results: List[Dict],
        filepath: Path,
    ) -> None:
        """Generate table extraction results table."""
        rows = []
        for result in table_results:
            rows.append({
                'Model': result['model_name'],
                'Charts': str(result.get('num_evaluated_charts', 0)),
                'Cells': str(result.get('num_evaluated_cells', 0)),
                'MAE': LaTeXExporter.format_float(result.get('mae')),
                'Median AE': LaTeXExporter.format_float(result.get('median_ae')),
                'MRE': LaTeXExporter.format_float(result.get('mre')),
                'Accuracy': LaTeXExporter.format_float(result.get('accuracy_with_tolerance')),
                'Exact Match': LaTeXExporter.format_float(result.get('exact_match_rate')),
            })
        
        df = pd.DataFrame(rows)
        
        LaTeXExporter.dataframe_to_latex(
            df,
            caption='Chart-to-Table Extraction Results',
            label='table_extraction_overall',
            filepath=filepath,
        )
    
    @staticmethod
    def generate_verification_table(
        verification_results: List[Dict],
        filepath: Path,
    ) -> None:
        """Generate verification results table."""
        rows = []
        for result in verification_results:
            rows.append({
                'Model': result['model_name'],
                'Evaluated': str(result.get('evaluated_claims', 0)),
                'Accuracy': LaTeXExporter.format_float(result.get('accuracy')),
                'Macro F1': LaTeXExporter.format_float(result.get('macro_f1')),
                'Avg Grounding': LaTeXExporter.format_float(result.get('average_grounding')),
            })
        
        df = pd.DataFrame(rows)
        
        LaTeXExporter.dataframe_to_latex(
            df,
            caption='Claim Verification Results',
            label='verification_overall',
            filepath=filepath,
        )
