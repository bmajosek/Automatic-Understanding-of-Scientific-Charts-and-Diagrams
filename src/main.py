"""
Main CLI interface for chart understanding evaluation.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import zipfile
import tarfile
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_project_config
from io_utils import create_csv_template, save_csv, validate_csv_schema
from evaluation import Evaluator
from plots import PlotGenerator
from latex_export import LaTeXExporter
from report_generator import ReportGenerator
from dataset_registry import DatasetRegistry
from dataset_catalog import list_downloadable, get_entry, huggingface_repo_id
from dataset_merge import save_staging, merge_staging_into_processed
from dataset_converters import get_converter
from experiments_report import generate_experiments_chapter
from thesis_plots import generate_all_thesis_figures


class EvaluationProject:
    """Main project manager."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize project."""
        default_cfg = Path('config/config.yaml')
        cfg_file = Path(config_path) if config_path else default_cfg
        if cfg_file.exists():
            proj = load_project_config(cfg_file, project_root=Path.cwd())
            self.config = proj.config
            self.data_dir = proj.data_dir
            self.raw_data_dir = proj.raw_data_dir
            self.processed_data_dir = proj.processed_data_dir
            self.predictions_dir = proj.predictions_dir
            self.results_dir = proj.results_dir
            self.images_dir = Path(self.config.get('paths', {}).get('images_dir', 'data/images'))
        else:
            self.config = self._default_config()
            paths = self.config.get('paths', {})
            self.data_dir = Path(paths.get('data_dir', 'data/processed'))
            self.raw_data_dir = Path(paths.get('raw_data_dir', 'data/raw'))
            self.processed_data_dir = Path(paths.get('processed_data_dir', 'data/processed'))
            self.predictions_dir = Path(paths.get('predictions_dir', 'predictions'))
            self.results_dir = Path(paths.get('results_dir', 'results'))
            self.images_dir = Path(paths.get('images_dir', 'data/images'))
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default configuration."""
        return {
            'evaluation': {
                'numerical_tolerance': 0.05,
                'iou_threshold': 0.5,
                'use_test_split_only': False,
                'generate_latex': True,
                'generate_plots': True,
                'generate_report': True,
            },
            'paths': {
                'data_dir': 'data',
                'raw_data_dir': 'data/raw',
                'processed_data_dir': 'data/processed',
                'predictions_dir': 'predictions',
                'results_dir': 'results',
                'images_dir': 'data/images',
            },
        }
    
    def init_templates(self) -> None:
        """Initialize project templates."""
        print("Initializing project templates...")
        
        # Create directories
        for d in [self.data_dir, self.predictions_dir, self.results_dir, self.raw_data_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create CSV templates
        templates = {
            self.data_dir / 'charts.csv': [
                'chart_id', 'image_path', 'chart_type', 'source', 'split', 'width', 'height', 'notes'
            ],
            self.data_dir / 'tables_gt.csv': [
                'chart_id', 'series', 'category', 'value'
            ],
            self.data_dir / 'questions.csv': [
                'question_id', 'chart_id', 'split', 'task', 'operation', 'question', 'answer', 'answer_type',
                'paraphrase_group', 'dialogue_id', 'turn_id', 'source_dataset', 'language'
            ],
            self.data_dir / 'claims.csv': [
                'claim_id', 'chart_id', 'split', 'claim', 'label', 'notes', 'source_dataset'
            ],
            self.data_dir / 'summaries.csv': [
                'chart_id', 'split', 'summary', 'source_dataset', 'notes'
            ],
            self.data_dir / 'components_gt.csv': [
                'chart_id', 'component_id', 'component_type', 'x1', 'y1', 'x2', 'y2', 'text', 'source_dataset'
            ],
            self.data_dir / 'models.csv': [
                'model_name', 'model_family', 'approach', 'temperature', 'max_tokens', 'prompt_version', 'input_type', 'notes'
            ],
        }
        
        for filepath, columns in templates.items():
            if not filepath.exists():
                create_csv_template(columns, filepath)
                print(f"  Created: {filepath}")
        
        # Create example toy dataset
        self._create_toy_dataset()
        
        print("Templates created successfully!")
    
    def _create_toy_dataset(self) -> None:
        """Create a simple toy dataset for testing."""
        examples_dir = Path("examples/toy_dataset")
        examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Create toy charts CSV
        charts_data = [
            {'chart_id': 'toy_001', 'image_path': 'examples/toy_dataset/chart1.png', 'chart_type': 'bar',
             'source': 'toy', 'split': 'test', 'width': 800, 'height': 600, 'notes': 'Toy example'},
            {'chart_id': 'toy_002', 'image_path': 'examples/toy_dataset/chart2.png', 'chart_type': 'line',
             'source': 'toy', 'split': 'test', 'width': 800, 'height': 600, 'notes': 'Toy example'},
        ]
        pd.DataFrame(charts_data).to_csv(examples_dir / 'charts.csv', index=False)
        
        # Create toy tables GT
        tables_data = [
            {'chart_id': 'toy_001', 'series': 'Sales', 'category': 'Q1', 'value': '100'},
            {'chart_id': 'toy_001', 'series': 'Sales', 'category': 'Q2', 'value': '150'},
            {'chart_id': 'toy_002', 'series': 'Growth', 'category': 'Jan', 'value': '10'},
            {'chart_id': 'toy_002', 'series': 'Growth', 'category': 'Feb', 'value': '12'},
        ]
        pd.DataFrame(tables_data).to_csv(examples_dir / 'tables_gt.csv', index=False)
        
        # Create toy questions
        questions_data = [
            {'question_id': 'q_001', 'chart_id': 'toy_001', 'split': 'test', 'task': 'value_retrieval',
             'operation': 'read_value', 'question': 'What is Q1 Sales?', 'answer': '100', 'answer_type': 'numeric',
             'paraphrase_group': None, 'dialogue_id': None, 'turn_id': None, 'source_dataset': 'toy', 'language': None},
            {'question_id': 'q_002', 'chart_id': 'toy_001', 'split': 'test', 'task': 'comparison',
             'operation': 'compare', 'question': 'Which quarter has higher sales, Q1 or Q2?', 'answer': 'Q2',
             'answer_type': 'categorical', 'paraphrase_group': None, 'dialogue_id': None, 'turn_id': None,
             'source_dataset': 'toy', 'language': None},
        ]
        pd.DataFrame(questions_data).to_csv(examples_dir / 'questions.csv', index=False)
        
        # Create toy claims
        claims_data = [
            {'claim_id': 'c_001', 'chart_id': 'toy_001', 'split': 'test', 'claim': 'Q2 Sales are higher than Q1 Sales',
             'label': 'supported', 'notes': None, 'source_dataset': 'toy'},
        ]
        pd.DataFrame(claims_data).to_csv(examples_dir / 'claims.csv', index=False)
        
        print("  Created toy dataset in examples/toy_dataset/ (no toy_model predictions)")
    
    def validate_data(self) -> None:
        """Validate input data files."""
        print("Validating data files...")
        
        errors = []
        
        # Validate charts
        is_valid, errs = validate_csv_schema(
            self.data_dir / 'charts.csv',
            ['chart_id', 'image_path', 'chart_type', 'source', 'split'],
            'charts.csv'
        )
        errors.extend(errs)
        
        # Validate tables_gt
        if (self.data_dir / 'tables_gt.csv').exists():
            is_valid, errs = validate_csv_schema(
                self.data_dir / 'tables_gt.csv',
                ['chart_id', 'value'],
                'tables_gt.csv'
            )
            errors.extend(errs)
        
        # Validate questions
        if (self.data_dir / 'questions.csv').exists():
            is_valid, errs = validate_csv_schema(
                self.data_dir / 'questions.csv',
                ['question_id', 'chart_id', 'question', 'answer'],
                'questions.csv'
            )
            errors.extend(errs)
        
        # Validate claims
        if (self.data_dir / 'claims.csv').exists():
            is_valid, errs = validate_csv_schema(
                self.data_dir / 'claims.csv',
                ['claim_id', 'chart_id', 'claim', 'label'],
                'claims.csv'
            )
            errors.extend(errs)

        # Validate summaries
        if (self.data_dir / 'summaries.csv').exists():
            is_valid, errs = validate_csv_schema(
                self.data_dir / 'summaries.csv',
                ['chart_id', 'summary'],
                'summaries.csv'
            )
            errors.extend(errs)
        
        if errors:
            print("Validation errors found:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("  All validations passed!")
    
    def list_datasets(self) -> None:
        """List supported datasets."""
        print("Supported datasets:\n")
        
        for name in DatasetRegistry.list_datasets():
            meta = DatasetRegistry.get_dataset(name)
            if meta:
                auto = "auto-downloadable" if meta.can_download_auto else "manual download"
                print(f"- {meta.name} ({name})")
                print(f"  Source: {meta.source_type.value} ({auto})")
                print(f"  Tasks: {', '.join(meta.tasks or [])}")
                print(f"  URL: {meta.source_url}")
                print()
    
    @staticmethod
    def _huggingface_repo_id_from_url(url: Optional[str]) -> Optional[str]:
        """Extract a Hugging Face dataset repo id from a dataset URL."""
        if not url:
            return None

        cleaned_url = str(url).strip().rstrip('/')

        # Allow registry entries to store the repo id directly, e.g.
        # "ahmed-masry/ChartQA".
        if '://' not in cleaned_url and cleaned_url.count('/') == 1:
            return cleaned_url

        marker = 'huggingface.co/datasets/'
        if marker not in cleaned_url:
            return None

        repo_part = cleaned_url.split(marker, 1)[1]
        repo_part = repo_part.split('?', 1)[0].split('#', 1)[0].strip('/')
        pieces = [p for p in repo_part.split('/') if p]

        if len(pieces) < 2:
            return None

        return '/'.join(pieces[:2])

    @staticmethod
    def _archive_extract_dir(archive_path: Path) -> Path:
        """Return a clean extraction directory for common archive suffixes."""
        name = archive_path.name
        for suffix in ('.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz'):
            if name.endswith(suffix):
                return archive_path.parent / name[:-len(suffix)]
        return archive_path.with_suffix('')

    @staticmethod
    def _ensure_safe_member_path(destination_dir: Path, member_name: str) -> None:
        """Prevent archive members from writing outside destination_dir."""
        destination_root = destination_dir.resolve()
        member_path = (destination_dir / member_name).resolve()
        try:
            member_path.relative_to(destination_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Unsafe archive member path detected: {member_name}"
            ) from exc

    @classmethod
    def _safe_extract_zip(cls, archive_path: Path, destination_dir: Path) -> None:
        """Safely extract a zip file."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, 'r') as zip_file:
            for member in zip_file.infolist():
                cls._ensure_safe_member_path(destination_dir, member.filename)
            zip_file.extractall(destination_dir)

    @classmethod
    def _safe_extract_tar(cls, archive_path: Path, destination_dir: Path) -> None:
        """Safely extract a tar file."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, 'r:*') as tar_file:
            for member in tar_file.getmembers():
                cls._ensure_safe_member_path(destination_dir, member.name)
            # ``filter='data'`` rejects links and special files that could
            # otherwise escape the destination despite the member-name check.
            tar_file.extractall(destination_dir, filter="data")

    def _extract_archives(self, root_dir: Path) -> None:
        """Extract zip/tar archives found under root_dir."""
        archive_patterns = ('*.zip', '*.tar', '*.tar.gz', '*.tgz', '*.tar.bz2', '*.tbz2', '*.tar.xz', '*.txz')
        archives = []
        for pattern in archive_patterns:
            archives.extend(root_dir.rglob(pattern))

        archives = sorted(set(archives))
        if not archives:
            print("  No archives found to extract")
            return

        for archive_path in archives:
            destination_dir = self._archive_extract_dir(archive_path)

            if destination_dir.exists() and any(destination_dir.iterdir()):
                print(f"  Skipping already extracted archive: {archive_path.name}")
                continue

            print(f"  Extracting {archive_path.name} -> {destination_dir}")
            try:
                if archive_path.suffix.lower() == '.zip':
                    self._safe_extract_zip(archive_path, destination_dir)
                else:
                    self._safe_extract_tar(archive_path, destination_dir)
            except Exception as exc:
                print(f"  Warning: failed to extract {archive_path}: {exc}")

    def _download_huggingface_dataset(
        self,
        ds_name: str,
        repo_id: str,
        force_download: bool = False,
        extract_archives: bool = True,
    ) -> None:
        """Download a Hugging Face dataset repository into data/raw/<dataset>."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Install it with: "
                "pip install -U huggingface_hub"
            ) from exc

        target_dir = self.raw_data_dir / ds_name
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading Hugging Face dataset: {repo_id}")
        print(f"  Target directory: {target_dir}")

        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',
                local_dir=str(target_dir),
                force_download=force_download,
            )
        except Exception as exc:
            print(f"  Download failed for {ds_name}: {exc}")
            print("  If this is a gated/private dataset, run `huggingface-cli login` and try again.")
            return

        print(f"  Download complete: {local_path}")

        if extract_archives:
            self._extract_archives(target_dir)

    def download_data(
        self,
        dataset_name: Optional[str] = None,
        all_datasets: bool = False,
        force_download: bool = False,
        extract_archives: bool = True,
    ) -> None:
        """Download dataset files into the configured raw data directory."""
        if all_datasets:
            datasets = list_downloadable()
        elif dataset_name:
            datasets = [dataset_name]
        else:
            print("Please specify --dataset or --all")
            return

        fallback_hf_repos = {
            # Keeps ChartQA working even if the registry stores only a display URL.
            'chartqa': 'ahmed-masry/ChartQA',
        }

        for ds_name in datasets:
            print(f"Attempting to download {ds_name}...")
            meta = DatasetRegistry.get_dataset(ds_name)
            entry = get_entry(ds_name)

            can_dl = (entry and entry.get("can_download_auto")) or (
                meta and meta.can_download_auto
            )
            if not can_dl:
                print(f"  {ds_name} cannot be auto-downloaded")
                instr = (entry or {}).get("instructions") or (
                    getattr(meta, "download_instructions", None) if meta else None
                )
                if instr:
                    print(f"  Instructions: {instr}")
                continue

            repo_id = None
            if entry:
                repo_id = entry.get("hf_repo") or huggingface_repo_id(entry.get("source_url"))
            if not repo_id and meta:
                repo_id = self._huggingface_repo_id_from_url(getattr(meta, "source_url", None))
            repo_id = repo_id or fallback_hf_repos.get(ds_name.lower())

            if repo_id:
                self._download_huggingface_dataset(
                    ds_name=ds_name,
                    repo_id=repo_id,
                    force_download=force_download,
                    extract_archives=extract_archives,
                )
                continue

            print(f"  Automatic download is not implemented for source: {meta.source_url}")
    
    def prepare_data(self, dataset_name: Optional[str] = None, all_datasets: bool = False) -> None:
        """Prepare dataset by converting to standard format."""
        if all_datasets:
            from dataset_catalog import list_implemented
            datasets = list_implemented()
        elif dataset_name:
            datasets = [dataset_name]
        else:
            print("Please specify --dataset or --all")
            return

        max_examples = self.config.get("datasets", {}).get("max_examples_per_dataset")
        
        for ds_name in datasets:
            print(f"Preparing {ds_name}...")
            entry = get_entry(ds_name) or {}
            limit = entry.get("max_examples") or max_examples

            staging_out = self.processed_data_dir
            converter = get_converter(
                ds_name, self.raw_data_dir, staging_out, max_examples=limit,
            )
            if not converter:
                print(f"  No converter for {ds_name}")
                continue
            
            if not converter.check_raw_data():
                print(f"  Raw data for {ds_name} not found. Run: download-data --dataset {ds_name}")
                continue
            
            try:
                output_dfs = converter.convert()
                counts = {k: len(v) if v is not None and not v.empty else 0 for k, v in output_dfs.items()}
                if counts.get("charts", 0) == 0 and counts.get("questions", 0) == 0:
                    raise ValueError(f"Converter for {ds_name} produced empty charts/questions.")
                save_staging(output_dfs, self.processed_data_dir, ds_name)
                print(f"Dataset: {ds_name}")
                print(f"  charts: {counts.get('charts', 0)}")
                print(f"  questions: {counts.get('questions', 0)}")
                print(f"  tables_gt: {counts.get('tables_gt', 0)}")
                print(f"  claims: {counts.get('claims', 0)}")
                print(f"  components_gt: {counts.get('components_gt', 0)}")
                print(f"  summaries: {counts.get('summaries', 0)}")
            except NotImplementedError as e:
                print(f"  Not implemented: {e}")
            except Exception as e:
                print(f"  Error during conversion: {e}")

        if len(datasets) >= 1:
            print("Merging staged datasets into data/processed/...")
            merged = merge_staging_into_processed(self.processed_data_dir)
            for key, n in merged.items():
                if n:
                    print(f"  merged {key}: {n}")
    
    def run_evaluation(self, task: Optional[str] = None) -> None:
        """Run full evaluation."""
        print("Running evaluation...")
        
        evaluator = Evaluator(
            self.data_dir,
            self.predictions_dir,
            self.results_dir,
            self.config,
        )
        
        results = evaluator.run_full_evaluation()
        
        # Generate CSVs
        print("Generating output CSV files...")
        
        if results.get('table_extraction'):
            df = pd.DataFrame(results['table_extraction'])
            save_csv(df, self.results_dir / 'intermediate' / 'table_extraction_overall.csv')
        
        if results.get('qa'):
            df = pd.DataFrame(results['qa'])
            save_csv(df, self.results_dir / 'intermediate' / 'qa_overall.csv')
        
        if results.get('verification'):
            df = pd.DataFrame(results['verification'])
            save_csv(df, self.results_dir / 'intermediate' / 'verification_overall.csv')
        
        if results.get('summarization'):
            df = pd.DataFrame(results['summarization'])
            save_csv(df, self.results_dir / 'intermediate' / 'summarization_overall.csv')
            
        # Generate plots
        if self.config.get('evaluation', {}).get('generate_plots', True):
            print("Generating plots...")
            PlotGenerator.examples_per_dataset(
                evaluator.charts,
                self.results_dir / 'figures' / 'examples_per_dataset.png'
            )
            PlotGenerator.chart_type_distribution(
                evaluator.charts,
                self.results_dir / 'figures' / 'chart_type_distribution.png'
            )
            if not evaluator.questions.empty:
                PlotGenerator.question_task_distribution(
                    evaluator.questions,
                    self.results_dir / 'figures' / 'question_task_distribution.png'
                )
            if results.get('qa'):
                PlotGenerator.qa_accuracy_by_task(
                    results['qa'],
                    self.results_dir / 'figures' / 'qa_accuracy_by_task.png'
                )
                
            if results.get('verification'):
                PlotGenerator.verification_accuracy_by_model(
                    results['verification'],
                    self.results_dir / 'figures' / 'verification_accuracy_by_model.png'
                )
                PlotGenerator.verification_error_types(
                    results['verification'],
                    self.results_dir / 'figures' / 'verification_error_types.png'
                )

            if results.get('summarization'):
                PlotGenerator.summarization_f1_by_model(
                    results['summarization'],
                    self.results_dir / 'figures' / 'summarization_f1_by_model.png'
                )
                PlotGenerator.summarization_length_by_model(
                    results['summarization'],
                    self.results_dir / 'figures' / 'summarization_length_by_model.png'
                )
            thesis_paths = generate_all_thesis_figures(
                evaluator.charts,
                evaluator.questions,
                results.get('qa', []),
                self.results_dir / 'figures',
                claims=evaluator.claims,
                verification_results=results.get('verification', []),
            )
            for p in thesis_paths:
                print(f"  Thesis figure: {p}")
        
        # Generate LaTeX tables
        if self.config.get('evaluation', {}).get('generate_latex', True):
            print("Generating LaTeX tables...")
            if results.get('table_extraction'):
                LaTeXExporter.generate_table_extraction_table(
                    results['table_extraction'],
                    self.results_dir / 'tables' / 'table_extraction_overall.tex'
                )
            if results.get('qa'):
                LaTeXExporter.generate_qa_table(
                    results['qa'],
                    self.results_dir / 'tables' / 'qa_overall.tex'
                )
            if results.get('verification'):
                LaTeXExporter.generate_verification_table(
                    results['verification'],
                    self.results_dir / 'tables' / 'verification_overall.tex'
                )
        
        # Generate report
        if self.config.get('evaluation', {}).get('generate_report', True):
            print("Generating report...")
            report = ReportGenerator.generate_full_report(
                evaluator.charts,
                evaluator.questions,
                evaluator.claims,
                evaluator.components_gt,
                evaluator.models,
                results.get('qa', []),
                results.get('table_extraction', []),
                results.get('verification', []),
            )
            report_path = self.results_dir / 'report' / 'results_chapter_draft.md'
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"  Report saved to {report_path}")

            exp_report = generate_experiments_chapter(
                evaluator.charts,
                evaluator.questions,
                evaluator.models,
                self.processed_data_dir,
            )
            exp_path = self.results_dir / 'report' / 'experiments_chapter_draft.md'
            with open(exp_path, 'w', encoding='utf-8') as f:
                f.write(exp_report)
            print(f"  Experiments chapter: {exp_path}")
        
        print("Evaluation complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Chart Understanding Evaluation Pipeline'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Init command
    subparsers.add_parser('init', help='Initialize project templates')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate input files')
    
    # List datasets command
    subparsers.add_parser('list-datasets', help='List supported datasets')
    
    # Download data command
    download_parser = subparsers.add_parser('download-data', help='Download dataset')
    download_parser.add_argument('--dataset', help='Dataset name')
    download_parser.add_argument('--all', action='store_true', help='Download all datasets')
    download_parser.add_argument('--force', action='store_true', help='Force re-download from the remote source')
    download_parser.add_argument('--no-extract', action='store_true', help='Do not extract downloaded zip/tar archives')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare-data', help='Prepare dataset')
    prepare_parser.add_argument('--dataset', help='Dataset name')
    prepare_parser.add_argument('--all', action='store_true', help='Prepare all datasets')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    eval_parser.add_argument('--task', help='Specific task to evaluate')
    
    args = parser.parse_args()
    
    # Create project
    config_path = getattr(args, 'config', None)
    project = EvaluationProject(Path(config_path) if config_path else None)
    
    # Execute command
    if args.command == 'init':
        project.init_templates()
    elif args.command == 'validate':
        project.validate_data()
    elif args.command == 'list-datasets':
        project.list_datasets()
    elif args.command == 'download-data':
        project.download_data(
            args.dataset,
            args.all,
            force_download=getattr(args, 'force', False),
            extract_archives=not getattr(args, 'no_extract', False),
        )
    elif args.command == 'prepare-data':
        project.prepare_data(args.dataset, args.all)
    elif args.command == 'evaluate':
        project.run_evaluation(getattr(args, 'task', None))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
