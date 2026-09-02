"""
Dataset registry and metadata management.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class SourceType(str, Enum):
    """Dataset source types."""
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    GITHUB = "github"
    OFFICIAL_PAGE = "official_page"
    MANUAL = "manual"


@dataclass
class DatasetMetadata:
    """Dataset metadata."""
    name: str
    source_type: SourceType
    source_url: Optional[str] = None
    version: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    tasks: List[str] = None
    chart_types: List[str] = None
    has_image: bool = True
    can_download_auto: bool = True
    download_instructions: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'source_type': self.source_type.value,
            'source_url': self.source_url,
            'version': self.version,
            'license': self.license,
            'citation': self.citation,
            'tasks': self.tasks or [],
            'chart_types': self.chart_types or [],
            'has_image': self.has_image,
            'can_download_auto': self.can_download_auto,
            'download_instructions': self.download_instructions,
        }


class DatasetRegistry:
    """Registry of supported datasets."""
    
    DATASETS = {
        'figureqa': DatasetMetadata(
            name='FigureQA',
            source_type=SourceType.OFFICIAL_PAGE,
            source_url='https://www.microsoft.com/en-us/research/project/figureqa/',
            version='1.0',
            license='CC0',
            tasks=['visual_reasoning', 'qa'],
            chart_types=['bar', 'line', 'pie'],
            can_download_auto=False,
            download_instructions='Download from official page or use datasets library.',
        ),
        'dvqa': DatasetMetadata(
            name='DVQA',
            source_type=SourceType.OFFICIAL_PAGE,
            source_url='https://github.com/chiragii/DVQA',
            version='1.0',
            license='Apache 2.0',
            tasks=['value_retrieval', 'comparison', 'arithmetic'],
            chart_types=['bar', 'grouped_bar', 'stacked_bar'],
            can_download_auto=False,
        ),
        'plotqa': DatasetMetadata(
            name='PlotQA',
            source_type=SourceType.OFFICIAL_PAGE,
            source_url='https://github.com/bhavishadaswani/PlotQA',
            version='1.0',
            license='CC0',
            tasks=['arithmetic', 'comparison', 'reasoning'],
            chart_types=['line', 'scatter', 'bar'],
            can_download_auto=False,
        ),
        'chartqa': DatasetMetadata(
            name='ChartQA',
            source_type=SourceType.HUGGINGFACE,
            source_url='https://huggingface.co/datasets/ahmed-masry/ChartQA',
            version='1.0',
            license='CC-BY-4.0',
            tasks=['qa', 'reasoning'],
            chart_types=['bar', 'line', 'pie', 'scatter'],
            can_download_auto=True,
        ),
        'chartcheck': DatasetMetadata(
            name='ChartCheck',
            source_type=SourceType.OFFICIAL_PAGE,
            source_url='https://github.com/allenai/chartcheck',
            version='1.0',
            license='Apache 2.0',
            tasks=['verification', 'reasoning'],
            chart_types=['bar', 'line'],
            can_download_auto=False,
        ),
        'scigraphqa': DatasetMetadata(
            name='SciGraphQA',
            source_type=SourceType.HUGGINGFACE,
            source_url='https://huggingface.co/datasets/Minseong/SciGraphQA',
            version='1.0',
            license='CC-BY-4.0',
            tasks=['multi_turn', 'reasoning', 'dialogue'],
            chart_types=['line', 'scatter', 'bar'],
            can_download_auto=False,
        ),
        'chartbench': DatasetMetadata(
            name='ChartBench',
            source_type=SourceType.OFFICIAL_PAGE,
            source_url='https://github.com/vistext/ChartBench',
            version='1.0',
            license='MIT',
            tasks=['reasoning', 'qa'],
            chart_types=['bar', 'line', 'pie', 'scatter', 'area'],
            can_download_auto=False,
        ),
        'chartx': DatasetMetadata(
            name='ChartX',
            source_type=SourceType.HUGGINGFACE,
            source_url='https://huggingface.co/datasets/THUDM/ChartX',
            version='1.0',
            license='CC-BY-4.0',
            tasks=['table_extraction', 'qa', 'verification', 'reasoning'],
            chart_types=['bar', 'line', 'pie', 'scatter', 'area', 'heatmap'],
            can_download_auto=False,
        ),
        'chartqapro': DatasetMetadata(
            name='ChartQAPro',
            source_type=SourceType.HUGGINGFACE,
            source_url='https://huggingface.co/datasets/ahmed-masry/ChartQAPro',
            version='1.0',
            license='CC-BY-4.0',
            tasks=['qa', 'reasoning', 'hypothetical'],
            chart_types=['bar', 'line', 'pie', 'scatter'],
            can_download_auto=False,
        ),
        'polychartqa': DatasetMetadata(
            name='PolyChartQA',
            source_type=SourceType.HUGGINGFACE,
            source_url='https://huggingface.co/datasets/kakaobrain/PolyChartQA',
            version='1.0',
            license='CC-BY-4.0',
            tasks=['qa', 'multilingual'],
            chart_types=['bar', 'line', 'pie'],
            can_download_auto=False,
        ),
        'chart-hqa': DatasetMetadata(
            name='Chart-HQA',
            source_type=SourceType.OFFICIAL_PAGE,
            source_url=None,
            version='1.0',
            license='unknown',
            tasks=['qa', 'reasoning'],
            chart_types=['bar', 'line', 'pie'],
            can_download_auto=False,
            download_instructions='Add raw data under data/raw/chart-hqa when available.',
        ),
        'chart_to_text': DatasetMetadata(
            name='Chart-to-Text',
            source_type=SourceType.OFFICIAL_PAGE,
            source_url='https://github.com/vis-nlp/ChartToText',
            version='1.0',
            license='CC-BY-4.0',
            tasks=['summarization'],
            chart_types=['bar', 'line', 'pie', 'scatter'],
            can_download_auto=False,
        ),
    }
    
    @classmethod
    def get_dataset(cls, name: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by name."""
        return cls.DATASETS.get(name.lower())
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available datasets."""
        return sorted(cls.DATASETS.keys())
    
    @classmethod
    def list_downloadable(cls) -> List[str]:
        """List datasets that can be auto-downloaded."""
        return [
            name for name, meta in cls.DATASETS.items()
            if meta.can_download_auto
        ]
    
    @classmethod
    def list_by_task(cls, task: str) -> List[str]:
        """List datasets supporting a specific task."""
        return [
            name for name, meta in cls.DATASETS.items()
            if task in (meta.tasks or [])
        ]


class DatasetManifest:
    """Manifest for downloaded/converted dataset."""
    
    @staticmethod
    def create(
        dataset_name: str,
        source_type: str,
        num_examples: int,
        source_url: Optional[str] = None,
        version: Optional[str] = None,
        license: Optional[str] = None,
        citation: Optional[str] = None,
        original_splits: Optional[List[str]] = None,
        checksum: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create manifest dictionary."""
        return {
            'dataset_name': dataset_name,
            'source_type': source_type,
            'source_url': source_url,
            'download_date': datetime.now().isoformat(),
            'dataset_version': version,
            'license': license,
            'citation_key': citation,
            'original_split_names': original_splits or [],
            'number_of_raw_examples': num_examples,
            'checksum': checksum,
            'notes': notes,
        }
    
    @staticmethod
    def save(manifest: Dict[str, Any], filepath: Path) -> None:
        """Save manifest to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    @staticmethod
    def load(filepath: Path) -> Optional[Dict[str, Any]]:
        """Load manifest from JSON file."""
        if not filepath.exists():
            return None
        with open(filepath, 'r') as f:
            return json.load(f)
