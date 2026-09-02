"""
Plot generation using matplotlib.
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class PlotGenerator:
    """Generate plots for results."""
    
    DPI = 300
    FIGSIZE = (10, 6)
    
    @staticmethod
    def ensure_dir(path: Path) -> None:
        """Ensure output directory exists."""
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_figure(fig, filepath: Path, dpi: int = DPI) -> None:
        """Save figure to file."""
        PlotGenerator.ensure_dir(filepath.parent)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def examples_per_dataset(
        charts_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Plot number of examples per dataset."""
        if charts_df.empty:
            return
        
        dataset_counts = charts_df.groupby('source').size().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        dataset_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Number of Charts per Dataset')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def chart_type_distribution(
        charts_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Plot chart type distribution."""
        if charts_df.empty:
            return
        
        chart_counts = charts_df['chart_type'].value_counts()
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        chart_counts.plot(kind='barh', ax=ax, color='coral')
        ax.set_title('Distribution of Chart Types')
        ax.set_xlabel('Count')
        ax.grid(axis='x', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def chart_types_by_dataset(
        charts_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Plot chart type distribution per dataset."""
        if charts_df.empty:
            return
        
        cross = pd.crosstab(charts_df['source'], charts_df['chart_type'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        cross.plot(kind='bar', ax=ax, stacked=False)
        ax.set_title('Chart Type Distribution by Dataset')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Count')
        ax.legend(title='Chart Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def question_task_distribution(
        questions_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Plot question task distribution."""
        if questions_df.empty:
            return
        
        task_counts = questions_df['task'].value_counts()
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        task_counts.plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_title('Distribution of Question Tasks')
        ax.set_xlabel('Count')
        ax.grid(axis='x', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def overall_model_comparison(
        qa_results: List[Dict],
        table_results: List[Dict],
        output_path: Path,
    ) -> None:
        """Compare models across tasks."""
        if not qa_results and not table_results:
            return
        
        models = set()
        qa_acc = {}
        table_acc = {}
        
        for result in qa_results:
            models.add(result['model_name'])
            qa_acc[result['model_name']] = result.get('accuracy', 0.0)
        
        for result in table_results:
            models.add(result['model_name'])
            table_acc[result['model_name']] = result.get('accuracy_with_tolerance', 0.0)
        
        models = sorted(models)
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        x = np.arange(len(models))
        width = 0.35
        
        qa_vals = [qa_acc.get(m, 0.0) for m in models]
        table_vals = [table_acc.get(m, 0.0) for m in models]
        
        ax.bar(x - width/2, qa_vals, width, label='QA Accuracy', color='steelblue')
        ax.bar(x + width/2, table_vals, width, label='Table Accuracy', color='coral')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Overall Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def table_relative_error_boxplot(
        table_results: List[Dict],
        output_path: Path,
    ) -> None:
        """Boxplot of relative errors per model."""
        if not table_results:
            return
        
        data = []
        labels = []
        
        for result in table_results:
            mre = result.get('mre')
            if not np.isnan(mre):
                data.append([mre])
                labels.append(result['model_name'])
        
        if not data:
            return
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        ax.boxplot(data, labels=labels)
        ax.set_ylabel('Mean Relative Error')
        ax.set_title('Table Extraction: Relative Error Distribution')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def verification_accuracy_by_model(
        verification_results: List[Dict],
        output_path: Path,
    ) -> None:
        if not verification_results:
            return

        models = []
        values = []

        for result in verification_results:
            model = result.get("model_name")
            acc = result.get("accuracy", result.get("label_accuracy", None))
            if model and acc is not None and not pd.isna(acc):
                models.append(model)
                values.append(float(acc))

        if not models:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(models, values)
        ax.set_title("Verification Accuracy by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        PlotGenerator.save_figure(fig, output_path)


    @staticmethod
    def verification_error_types(
        verification_results: List[Dict],
        output_path: Path,
    ) -> None:
        if not verification_results:
            return

        all_errors = {}

        for result in verification_results:
            for error_type, count in result.get("error_type_distribution", {}).items():
                key = str(error_type)[:80]
                all_errors[key] = all_errors.get(key, 0) + int(count)

        if not all_errors:
            return

        errors = sorted(all_errors, key=all_errors.get, reverse=True)[:20]
        counts = [all_errors[e] for e in errors]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(errors, counts)
        ax.set_title("Verification Error Type Distribution")
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        PlotGenerator.save_figure(fig, output_path)


    @staticmethod
    def summarization_f1_by_model(
        summarization_results: List[Dict],
        output_path: Path,
    ) -> None:
        if not summarization_results:
            return

        models = []
        values = []

        for result in summarization_results:
            model = result.get("model_name")
            f1 = result.get("avg_word_overlap_f1", None)
            if model and f1 is not None and not pd.isna(f1):
                models.append(model)
                values.append(float(f1))

        if not models:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(models, values)
        ax.set_title("Summarization Word-Overlap F1 by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Word-Overlap F1")
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        PlotGenerator.save_figure(fig, output_path)


    @staticmethod
    def summarization_length_by_model(
        summarization_results: List[Dict],
        output_path: Path,
    ) -> None:
        if not summarization_results:
            return

        models = []
        pred_lengths = []
        gt_lengths = []

        for result in summarization_results:
            model = result.get("model_name")
            pred_len = result.get("avg_pred_length", None)
            gt_len = result.get("avg_gt_length", None)

            if model and pred_len is not None and gt_len is not None:
                if not pd.isna(pred_len) and not pd.isna(gt_len):
                    models.append(model)
                    pred_lengths.append(float(pred_len))
                    gt_lengths.append(float(gt_len))

        if not models:
            return

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, pred_lengths, width, label="Predicted summary")
        ax.bar(x + width / 2, gt_lengths, width, label="Ground truth summary")

        ax.set_title("Average Summary Length by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Word Count")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        PlotGenerator.save_figure(fig, output_path)
        
    @staticmethod
    def qa_accuracy_by_task(
        qa_results: List[Dict],
        output_path: Path,
    ) -> None:
        """Plot QA accuracy by task."""
        if not qa_results:
            return
        
        # Aggregate across models
        task_acc = {}
        for result in qa_results:
            for task, acc in result.get('accuracy_by_task', {}).items():
                if isinstance(acc, dict):
                    total = acc.get("total", 0)
                    acc = acc.get("correct", 0) / total if total else 0.0
                if task not in task_acc:
                    task_acc[task] = []
                task_acc[task].append(acc)
        
        avg_acc = {task: np.mean(accs) for task, accs in task_acc.items()}
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        tasks = sorted(avg_acc.keys())
        accs = [avg_acc[t] for t in tasks]
        
        ax.bar(tasks, accs, color='lightblue')
        ax.set_xlabel('Task')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('QA Accuracy by Task Type')
        ax.set_ylim([0, 1.0])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def qa_accuracy_by_operation(
        qa_results: List[Dict],
        output_path: Path,
    ) -> None:
        """Plot QA accuracy by operation."""
        if not qa_results:
            return
        
        op_acc = {}
        for result in qa_results:
            for op, acc in result.get('accuracy_by_operation', {}).items():
                if isinstance(acc, dict):
                    total = acc.get("total", 0)
                    acc = acc.get("correct", 0) / total if total else 0.0
                if op not in op_acc:
                    op_acc[op] = []
                op_acc[op].append(acc)
        
        avg_acc = {op: np.mean(accs) for op, accs in op_acc.items()}
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        ops = sorted(avg_acc.keys())
        accs = [avg_acc[o] for o in ops]
        
        ax.bar(ops, accs, color='lightcoral')
        ax.set_xlabel('Operation')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('QA Accuracy by Operation Type')
        ax.set_ylim([0, 1.0])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def qa_accuracy_by_dataset(
        qa_results: List[Dict],
        output_path: Path,
    ) -> None:
        """Plot QA accuracy by dataset."""
        if not qa_results:
            return
        
        ds_acc = {}
        for result in qa_results:
            for ds, acc in result.get('accuracy_by_dataset', {}).items():
                if ds not in ds_acc:
                    ds_acc[ds] = []
                ds_acc[ds].append(acc)
        
        avg_acc = {ds: np.mean(accs) for ds, accs in ds_acc.items()}
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        datasets = sorted(avg_acc.keys())
        accs = [avg_acc[d] for d in datasets]
        
        ax.bar(datasets, accs, color='lightgreen')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('QA Accuracy by Dataset')
        ax.set_ylim([0, 1.0])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def qa_error_types(
        qa_results: List[Dict],
        output_path: Path,
    ) -> None:
        """Plot error type distribution."""
        if not qa_results:
            return
        
        all_errors = {}
        for result in qa_results:
            for error_type, count in result.get('error_type_distribution', {}).items():
                all_errors[error_type] = all_errors.get(error_type, 0) + count
        
        if not all_errors:
            return
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        error_types = sorted(all_errors.keys())
        counts = [all_errors[e] for e in error_types]
        
        ax.barh(error_types, counts, color='salmon')
        ax.set_xlabel('Count')
        ax.set_title('Error Type Distribution')
        ax.grid(axis='x', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
    
    @staticmethod
    def explanation_grounding_by_model(
        verification_results: List[Dict],
        output_path: Path,
    ) -> None:
        """Plot explanation grounding by model."""
        if not verification_results:
            return
        
        models = []
        groundings = []
        
        for result in verification_results:
            grounding = result.get('average_grounding')
            if not np.isnan(grounding):
                models.append(result['model_name'])
                groundings.append(grounding)
        
        if not models:
            return
        
        fig, ax = plt.subplots(figsize=PlotGenerator.FIGSIZE)
        ax.bar(models, groundings, color='plum')
        ax.set_ylabel('Average Explanation Grounding')
        ax.set_title('Explanation Grounding by Model')
        ax.set_ylim([0, 1.0])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        PlotGenerator.save_figure(fig, output_path)
