"""
Core evaluation logic for chart understanding tasks.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from io_utils import load_csv
from model_registry import EXCLUDE_FROM_SCOREBOARD, scoreboard_models
from pipelines.common import is_prediction_failure
from metrics import (
    NumericalMetrics,
    AnswerEvaluation,
    ComponentMetrics,
    ConfusionMatrix,
)


class Evaluator:
    """Main evaluation orchestrator."""
    
    def __init__(
        self,
        data_dir: Path,
        predictions_dir: Path,
        results_dir: Path,
        config: Dict[str, Any],
    ):
        self.data_dir = Path(data_dir)
        self.predictions_dir = Path(predictions_dir)
        self.results_dir = Path(results_dir)
        self.config = config
        
        # Create output directories
        self.intermediate_dir = self.results_dir / "intermediate"
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir = self.results_dir / "figures"
        self.report_dir = self.results_dir / "report"
        
        for d in [self.intermediate_dir, self.tables_dir, self.figures_dir, self.report_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.charts = self._load_charts()
        self.tables_gt = self._load_tables_gt()
        self.questions = self._load_questions()
        self.claims = self._load_claims()
        self.summaries = self._load_summaries()
        self.components_gt = self._load_components_gt()
        self.models = self._load_models()
    
    def _load_charts(self) -> pd.DataFrame:
        """Load chart metadata."""
        path = self.data_dir / "charts.csv"
        if not path.exists():
            return pd.DataFrame()
        return load_csv(path)
    
    def _safe_load_csv(self, path: Path, columns: list[str]) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(columns=columns)

        try:
            return load_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=columns)


    def _load_summaries(self) -> pd.DataFrame:
        return self._safe_load_csv(
            self.data_dir / "summaries.csv",
            ["chart_id", "split", "summary", "source_dataset", "notes"],
        )
        
    def _load_tables_gt(self) -> pd.DataFrame:
        path = self.data_dir / "tables.csv"

        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(
                columns=["chart_id", "series", "category", "gt_value"]
            )

        try:
            return load_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(
                columns=["chart_id", "series", "category", "gt_value"]
            )
    
    def _load_questions(self) -> pd.DataFrame:
        """Load questions."""
        path = self.data_dir / "questions.csv"
        if not path.exists():
            return pd.DataFrame()
        return load_csv(path)
    
    def _load_claims(self) -> pd.DataFrame:
        """Load claims."""
        path = self.data_dir / "claims.csv"
        if not path.exists():
            return pd.DataFrame()
        return load_csv(path)
    
    def _load_components_gt(self) -> pd.DataFrame:
        path = self.data_dir / "components.csv"

        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(
                columns=["chart_id", "component_id", "component_type", "bbox"]
            )

        try:
            return load_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(
                columns=["chart_id", "component_id", "component_type", "bbox"]
            )
    
    def _load_models(self) -> pd.DataFrame:
        """Load model setup."""
        path = self.data_dir / "models.csv"
        if not path.exists():
            return pd.DataFrame()
        return load_csv(path)
    
    def evaluate_table_extraction(self, model_name: str) -> Dict[str, Any]:
        """Evaluate chart-to-table extraction."""
        results = {
            "model_name": model_name,
            "num_evaluated_charts": 0,
            "num_evaluated_cells": 0,
            "missing_predictions": 0,
            "mae": np.nan,
            "median_ae": np.nan,
            "mre": np.nan,
            "median_re": np.nan,
            "accuracy_with_tolerance": np.nan,
            "exact_match_rate": 0.0,
            "cell_level_results": [],
            "chart_level_results": [],
        }
        
        if self.tables_gt.empty:
            return results
        
        pred_path = self.predictions_dir / model_name / "tables_pred.csv"
        if not pred_path.exists():
            return results
        
        pred_tables = load_csv(pred_path)
        
        gt_values = []
        pred_values = []
        exact_matches = 0
        total_cells = 0
        
        for _, row in self.tables_gt.iterrows():
            chart_id = row['chart_id']
            series = row.get('series')
            category = row.get('category')
            gt_val = row.get('value')
            
            matching_pred = pred_tables[
                (pred_tables['chart_id'] == chart_id) &
                (pred_tables.get('series', '') == (series or '')) &
                (pred_tables.get('category', '') == (category or ''))
            ]
            
            if matching_pred.empty:
                results["missing_predictions"] += 1
            else:
                pred_val = matching_pred.iloc[0].get('pred_value')
                
                gt_num = NumericalMetrics.parse_number(str(gt_val))
                pred_num = NumericalMetrics.parse_number(str(pred_val))
                
                if gt_num is not None and pred_num is not None:
                    gt_values.append(gt_num)
                    pred_values.append(pred_num)
                    
                    if NumericalMetrics.parse_number(str(gt_val)) == NumericalMetrics.parse_number(str(pred_val)):
                        exact_matches += 1
                
                total_cells += 1
        
        if gt_values:
            results["num_evaluated_cells"] = len(gt_values)
            results["mae"] = NumericalMetrics.mae(gt_values, pred_values)
            results["median_ae"] = NumericalMetrics.median_ae(gt_values, pred_values)
            results["mre"] = NumericalMetrics.mre(gt_values, pred_values)
            results["median_re"] = NumericalMetrics.median_re(gt_values, pred_values)
            tolerance = self.config.get("evaluation", {}).get("numerical_tolerance", 0.05)
            results["accuracy_with_tolerance"] = NumericalMetrics.accuracy_with_tolerance(
                gt_values, pred_values, tolerance
            )
        
        if total_cells > 0:
            results["exact_match_rate"] = exact_matches / total_cells
        
        results["num_evaluated_charts"] = len(set(self.tables_gt['chart_id'].unique()))
        
        return results
    
    def evaluate_qa(self, model_name: str) -> Dict[str, Any]:
        """Evaluate chart question answering."""
        results = {
            "model_name": model_name,
            "total_questions": len(self.questions),
            "prediction_rows": 0,
            "valid_prediction_rows": 0,
            "failed_prediction_rows": 0,
            "attempted_coverage_of_gt": 0.0,
            "coverage_of_gt": 0.0,
            "is_pilot_run": False,
            "evaluated_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "accuracy_by_task": {},
            "samples_by_task": {},
            "accuracy_by_operation": {},
            "accuracy_by_dataset": {},
            "numerical_mae": np.nan,
            "numerical_mre": np.nan,
            "error_type_distribution": {},
            "row_level_results": [],
        }
        
        if self.questions.empty:
            return results
        
        pred_path = self.predictions_dir / model_name / "qa_pred.csv"
        if not pred_path.exists():
            return results
        
        pred_qa = load_csv(pred_path)
        results["prediction_rows"] = len(pred_qa)
        pred_qa = pred_qa.drop_duplicates(subset=["question_id"], keep="first")
        pred_lookup = pred_qa.set_index(
            pred_qa["question_id"].astype(str), drop=False
        )
        
        correct = 0
        evaluated = 0
        numerical_gt = []
        numerical_pred = []
        error_types = {}
        
        for _, q_row in self.questions.iterrows():
            q_id = q_row['question_id']
            gt_answer = q_row['answer']
            answer_type = q_row.get('answer_type', 'text')
            task = q_row.get('task', 'unknown')
            operation = q_row.get('operation', 'other')
            dataset = q_row.get('source_dataset', 'unknown')
            
            if str(q_id) not in pred_lookup.index:
                continue

            pred_row = pred_lookup.loc[str(q_id)]
            pred_answer = pred_row.get('pred_answer')
            error_type = pred_row.get('error_type')
            err_key = "ok"
            if error_type is not None and str(error_type).strip() and str(error_type).lower() != "nan":
                err_key = str(error_type).strip()[:120]
            error_types[err_key] = error_types.get(err_key, 0) + 1

            if is_prediction_failure(error_type):
                results["failed_prediction_rows"] += 1
                continue
            
            is_correct = AnswerEvaluation.evaluate_answer(
                gt_answer, pred_answer, answer_type,
                self.config.get("evaluation", {}).get("numerical_tolerance", 0.05)
            )
            
            evaluated += 1
            if is_correct:
                correct += 1
            
            if answer_type == "numeric":
                gt_num = NumericalMetrics.parse_number(str(gt_answer))
                pred_num = NumericalMetrics.parse_number(str(pred_answer))
                if gt_num is not None and pred_num is not None:
                    numerical_gt.append(gt_num)
                    numerical_pred.append(pred_num)
            
            # Task-level accuracy
            if task not in results["accuracy_by_task"]:
                results["accuracy_by_task"][task] = {"correct": 0, "total": 0}
            results["accuracy_by_task"][task]["total"] += 1
            if is_correct:
                results["accuracy_by_task"][task]["correct"] += 1
            
            # Operation-level accuracy
            if operation not in results["accuracy_by_operation"]:
                results["accuracy_by_operation"][operation] = {"correct": 0, "total": 0}
            results["accuracy_by_operation"][operation]["total"] += 1
            if is_correct:
                results["accuracy_by_operation"][operation]["correct"] += 1
            
            # Dataset-level accuracy
            if dataset not in results["accuracy_by_dataset"]:
                results["accuracy_by_dataset"][dataset] = {"correct": 0, "total": 0}
            results["accuracy_by_dataset"][dataset]["total"] += 1
            if is_correct:
                results["accuracy_by_dataset"][dataset]["correct"] += 1
        
        if evaluated > 0:
            results["evaluated_questions"] = evaluated
            results["valid_prediction_rows"] = evaluated
            results["correct_answers"] = correct
            results["accuracy"] = correct / evaluated
            total_gt = max(len(self.questions), 1)
            results["coverage_of_gt"] = evaluated / total_gt
            results["attempted_coverage_of_gt"] = (
                (evaluated + results["failed_prediction_rows"]) / total_gt
            )
            results["is_pilot_run"] = (
                len(pred_qa) < 500 or results["coverage_of_gt"] < 0.05
            )
        elif results["failed_prediction_rows"]:
            results["attempted_coverage_of_gt"] = (
                results["failed_prediction_rows"] / max(len(self.questions), 1)
            )

        # Compute per-category accuracies.
        for task_key in results["accuracy_by_task"]:
            counts = results["accuracy_by_task"][task_key]
            if isinstance(counts, dict):
                results["samples_by_task"][task_key] = counts["total"]
                results["accuracy_by_task"][task_key] = (
                    counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
                )

        for op_key in results["accuracy_by_operation"]:
            counts = results["accuracy_by_operation"][op_key]
            if isinstance(counts, dict):
                results["accuracy_by_operation"][op_key] = (
                    counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
                )

        for ds_key in results["accuracy_by_dataset"]:
            counts = results["accuracy_by_dataset"][ds_key]
            if isinstance(counts, dict):
                results["accuracy_by_dataset"][ds_key] = (
                    counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
                )
        
        if numerical_gt:
            results["numerical_mae"] = NumericalMetrics.mae(numerical_gt, numerical_pred)
            results["numerical_mre"] = NumericalMetrics.mre(numerical_gt, numerical_pred)
        
        results["error_type_distribution"] = error_types
        
        return results

    def evaluate_summarization(self, model_name: str) -> Dict[str, Any]:
        results = {
            "model_name": model_name,
            "total_summaries": len(self.summaries),
            "prediction_rows": 0,
            "evaluated_summaries": 0,
            "coverage_of_gt": 0.0,
            "avg_pred_length": 0.0,
            "avg_gt_length": 0.0,
            "avg_word_overlap_f1": 0.0,
            "error_type_distribution": {},
        }

        if self.summaries.empty:
            return results

        pred_path = self.predictions_dir / model_name / "summaries_pred.csv"
        if not pred_path.exists():
            return results

        pred_df = load_csv(pred_path)
        results["prediction_rows"] = len(pred_df)

        scores = []
        pred_lengths = []
        gt_lengths = []
        error_types = {}

        for _, gt_row in self.summaries.iterrows():
            chart_id = str(gt_row.get("chart_id", "")).strip()
            gt_summary = str(gt_row.get("summary", "") or "").strip()

            match = pred_df[pred_df["chart_id"].astype(str) == chart_id]
            if match.empty:
                continue

            pred_row = match.iloc[0]
            pred_summary = str(pred_row.get("pred_summary", "") or "").strip()

            error_type = pred_row.get("error_type")
            err_key = "ok"
            if error_type is not None and str(error_type).strip() and str(error_type).lower() != "nan":
                err_key = str(error_type).strip()[:120]
            error_types[err_key] = error_types.get(err_key, 0) + 1

            gt_words = set(gt_summary.lower().split())
            pred_words = set(pred_summary.lower().split())

            if gt_words and pred_words:
                overlap = len(gt_words & pred_words)
                precision = overlap / len(pred_words)
                recall = overlap / len(gt_words)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            else:
                f1 = 0.0

            scores.append(f1)
            pred_lengths.append(len(pred_summary.split()))
            gt_lengths.append(len(gt_summary.split()))

        if scores:
            results["evaluated_summaries"] = len(scores)
            results["coverage_of_gt"] = len(scores) / max(len(self.summaries), 1)
            results["avg_word_overlap_f1"] = float(np.mean(scores))
            results["avg_pred_length"] = float(np.mean(pred_lengths))
            results["avg_gt_length"] = float(np.mean(gt_lengths))

        results["error_type_distribution"] = error_types
        return results
    
    def evaluate_verification(self, model_name: str) -> Dict[str, Any]:
        """Evaluate factual verification."""
        results = {
            "model_name": model_name,
            "total_claims": len(self.claims),
            "evaluated_claims": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class_metrics": {},
            "confusion_matrix": None,
            "accuracy_by_dataset": {},
            "average_grounding": np.nan,
            "ground_truth_label_count": 0,
            "ground_truth_labels": {},
            "single_class_warning": "",
        }
        
        if self.claims.empty:
            return results

        label_counts = self.claims["label"].fillna("unverifiable").value_counts().to_dict()
        results["ground_truth_labels"] = label_counts
        results["ground_truth_label_count"] = len(label_counts)
        if len(label_counts) < 2:
            results["single_class_warning"] = (
                "Verification ground truth contains only one label; accuracy and macro-F1 "
                "do not measure class discrimination."
            )
        
        pred_path = self.predictions_dir / model_name / "claims_pred.csv"
        if not pred_path.exists():
            return results
        
        pred_claims = load_csv(pred_path)
        
        gt_labels = []
        pred_labels = []
        grounding_scores = []
        
        for _, c_row in self.claims.iterrows():
            c_id = c_row['claim_id']
            gt_label = c_row.get('label', 'unverifiable')
            dataset = c_row.get('source_dataset', 'unknown')
            
            matching_pred = pred_claims[pred_claims['claim_id'] == c_id]
            if matching_pred.empty:
                continue
            
            pred_label = matching_pred.iloc[0].get('pred_label', 'unverifiable')
            
            gt_labels.append(gt_label)
            pred_labels.append(pred_label)
            
            # Grounding score
            supported = matching_pred.iloc[0].get('supported_explanation_claims')
            total = matching_pred.iloc[0].get('total_explanation_claims')
            if supported is not None and total is not None and total > 0:
                grounding_scores.append(float(supported) / float(total))
            
            # Dataset-level accuracy
            if dataset not in results["accuracy_by_dataset"]:
                results["accuracy_by_dataset"][dataset] = {"correct": 0, "total": 0}
            results["accuracy_by_dataset"][dataset]["total"] += 1
            if gt_label == pred_label:
                results["accuracy_by_dataset"][dataset]["correct"] += 1
        
        if gt_labels:
            results["evaluated_claims"] = len(gt_labels)
            results["accuracy"] = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred) / len(gt_labels)
            
            # Keep the denominator fixed across models.  Inferring labels from
            # model outputs made macro-F1 incomparable when different systems
            # emitted one, two, or three labels.
            cm = ConfusionMatrix.compute(
                gt_labels,
                pred_labels,
                label_order=["supported", "contradicted", "unverifiable"],
            )
            results["confusion_matrix"] = cm.to_dict()
            results["macro_f1"] = float(ConfusionMatrix.macro_f1(cm))
            
            for label in cm.index:
                precision, recall, f1 = ConfusionMatrix.precision_recall_f1(cm, label)
                results["per_class_metrics"][label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            
            # Dataset-level accuracies
            for ds_key in results["accuracy_by_dataset"]:
                counts = results["accuracy_by_dataset"][ds_key]
                results["accuracy_by_dataset"][ds_key] = (
                    counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
                )
        
        if grounding_scores:
            results["average_grounding"] = np.mean(grounding_scores)
        
        return results
    
    def evaluate_components(self, model_name: str) -> Dict[str, Any]:
        """Evaluate component detection."""
        results = {
            "model_name": model_name,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou": 0.0,
            "total_gt_components": 0,
            "total_pred_components": 0,
            "num_matched": 0,
        }
        
        if self.components_gt.empty:
            return results
        
        pred_path = self.predictions_dir / model_name / "components_pred.csv"
        if not pred_path.exists():
            return results
        
        pred_components = load_csv(pred_path)
        iou_threshold = self.config.get("evaluation", {}).get("iou_threshold", 0.5)
        
        all_ious = []
        
        for chart_id in self.components_gt['chart_id'].unique():
            gt_comp = self.components_gt[self.components_gt['chart_id'] == chart_id]
            pred_comp = pred_components[pred_components['chart_id'] == chart_id]
            
            gt_boxes = [
                {
                    'x1': float(row['x1']),
                    'y1': float(row['y1']),
                    'x2': float(row['x2']),
                    'y2': float(row['y2']),
                }
                for _, row in gt_comp.iterrows()
            ]
            
            pred_boxes = [
                {
                    'x1': float(row['x1']),
                    'y1': float(row['y1']),
                    'x2': float(row['x2']),
                    'y2': float(row['y2']),
                }
                for _, row in pred_comp.iterrows()
            ]
            
            results["total_gt_components"] += len(gt_boxes)
            results["total_pred_components"] += len(pred_boxes)
            
            matched_gt, matched_pred, matched_ious = ComponentMetrics.greedy_match_iou(
                gt_boxes, pred_boxes, iou_threshold
            )
            
            results["num_matched"] += len(matched_gt)
            all_ious.extend(matched_ious)
        
        if results["total_pred_components"] > 0:
            results["precision"] = results["num_matched"] / results["total_pred_components"]
        
        if results["total_gt_components"] > 0:
            results["recall"] = results["num_matched"] / results["total_gt_components"]
        
        if results["precision"] + results["recall"] > 0:
            results["f1"] = (
                2 * results["precision"] * results["recall"] /
                (results["precision"] + results["recall"])
            )
        
        if all_ious:
            results["mean_iou"] = float(np.mean(all_ious))
        
        return results
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run evaluation for all models and tasks."""
        results = {
            "table_extraction": [],
            "qa": [],
            "verification": [],
            "summarization": [],
            "components": [],
        }
        
        pred_dirs = []
        if self.predictions_dir.exists():
            pred_dirs = sorted(
                p.name for p in self.predictions_dir.iterdir() if p.is_dir()
            )
        model_names = scoreboard_models(self.data_dir, pred_dirs)
        if not model_names:
            model_names = [
                n for n in pred_dirs if n not in EXCLUDE_FROM_SCOREBOARD
            ]
        
        for model_name in model_names:
            if (self.predictions_dir / model_name / "tables_pred.csv").exists():
                te = self.evaluate_table_extraction(model_name)
                if te.get("num_evaluated_cells", 0) > 0:
                    results["table_extraction"].append(te)
            
            if (self.predictions_dir / model_name / "qa_pred.csv").exists():
                qa = self.evaluate_qa(model_name)
                if qa.get("evaluated_questions", 0) > 0:
                    results["qa"].append(qa)
            
            if (self.predictions_dir / model_name / "claims_pred.csv").exists():
                results["verification"].append(self.evaluate_verification(model_name))

            if (self.predictions_dir / model_name / "summaries_pred.csv").exists():
                sm = self.evaluate_summarization(model_name)
                if sm.get("evaluated_summaries", 0) > 0:
                    results["summarization"].append(sm)
            
            
            if (self.predictions_dir / model_name / "components_pred.csv").exists():
                results["components"].append(self.evaluate_components(model_name))
        
        return results
