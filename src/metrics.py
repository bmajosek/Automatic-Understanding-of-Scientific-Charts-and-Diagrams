"""
Evaluation metrics for chart understanding tasks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class NumericalMetrics:
    """Metrics for numerical value extraction and QA."""
    
    @staticmethod
    def parse_number(s: str) -> Optional[float]:
        """
        Parse a number from text, handling various formats.
        
        Args:
            s: String that may contain a number
        
        Returns:
            Parsed float or None
        """
        if not isinstance(s, str):
            return None
        
        s = s.strip().lower()
        if not s:
            return None
        
        # Try direct conversion
        try:
            return float(s)
        except ValueError:
            pass
        
        # Try removing common separators
        for sep in [',', ' ', '%', '$']:
            try:
                return float(s.replace(sep, ''))
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def mae(gt_values: List[float], pred_values: List[float]) -> float:
        """Mean Absolute Error."""
        if not gt_values:
            return np.nan
        return float(np.mean(np.abs(np.array(gt_values) - np.array(pred_values))))
    
    @staticmethod
    def median_ae(gt_values: List[float], pred_values: List[float]) -> float:
        """Median Absolute Error."""
        if not gt_values:
            return np.nan
        return float(np.median(np.abs(np.array(gt_values) - np.array(pred_values))))
    
    @staticmethod
    def relative_error(gt_values: List[float], pred_values: List[float]) -> List[float]:
        """Relative error for each pair."""
        errors = []
        for gt, pred in zip(gt_values, pred_values):
            if gt == 0:
                errors.append(float('inf') if pred != 0 else 0.0)
            else:
                errors.append(abs(pred - gt) / abs(gt))
        return errors
    
    @staticmethod
    def mre(gt_values: List[float], pred_values: List[float]) -> float:
        """Mean Relative Error."""
        errors = NumericalMetrics.relative_error(gt_values, pred_values)
        finite_errors = [e for e in errors if np.isfinite(e)]
        return float(np.mean(finite_errors)) if finite_errors else np.nan
    
    @staticmethod
    def median_re(gt_values: List[float], pred_values: List[float]) -> float:
        """Median Relative Error."""
        errors = NumericalMetrics.relative_error(gt_values, pred_values)
        finite_errors = [e for e in errors if np.isfinite(e)]
        return float(np.median(finite_errors)) if finite_errors else np.nan
    
    @staticmethod
    def accuracy_with_tolerance(
        gt_values: List[float],
        pred_values: List[float],
        tolerance: float = 0.05,
    ) -> float:
        """
        Accuracy with relative tolerance.
        
        Args:
            gt_values: Ground truth values
            pred_values: Predicted values
            tolerance: Relative tolerance (default 5%)
        
        Returns:
            Accuracy (0-1)
        """
        if not gt_values:
            return np.nan
        
        correct = 0
        for gt, pred in zip(gt_values, pred_values):
            if gt == 0:
                if pred == 0:
                    correct += 1
            else:
                rel_err = abs(pred - gt) / abs(gt)
                if rel_err <= tolerance:
                    correct += 1
        
        return correct / len(gt_values)


class TextMetrics:
    """Metrics for text/categorical answers."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()
    
    @staticmethod
    def normalize_boolean(text: str) -> Optional[str]:
        """
        Normalize boolean answers.
        
        Returns:
            'yes' or 'no' or None
        """
        text = text.lower().strip()
        if text in {'yes', 'true', 'supported', '1'}:
            return 'yes'
        elif text in {'no', 'false', 'contradicted', '0'}:
            return 'no'
        return None
    
    @staticmethod
    def exact_match(gt: str, pred: str) -> bool:
        """Exact match after normalization."""
        return TextMetrics.normalize_text(gt) == TextMetrics.normalize_text(pred)


class AnswerEvaluation:
    """Unified answer evaluation."""
    
    @staticmethod
    def evaluate_answer(
        gt: str,
        pred: str,
        answer_type: str = "text",
        numerical_tolerance: float = 0.05,
    ) -> bool:
        """
        Evaluate a predicted answer.
        
        Args:
            gt: Ground truth answer
            pred: Predicted answer
            answer_type: Type of answer (numeric, categorical, boolean, text)
            numerical_tolerance: Tolerance for numeric answers
        
        Returns:
            Whether prediction is correct
        """
        if answer_type == "numeric":
            gt_num = NumericalMetrics.parse_number(str(gt))
            pred_num = NumericalMetrics.parse_number(str(pred))
            
            if gt_num is None or pred_num is None:
                return False
            
            rel_err = abs(pred_num - gt_num) / abs(gt_num) if gt_num != 0 else (pred_num == 0)
            if gt_num == 0:
                return pred_num == 0
            return rel_err <= numerical_tolerance
        
        elif answer_type == "boolean":
            gt_bool = TextMetrics.normalize_boolean(str(gt))
            pred_bool = TextMetrics.normalize_boolean(str(pred))
            return gt_bool is not None and gt_bool == pred_bool
        
        else:  # categorical or text
            return TextMetrics.exact_match(str(gt), str(pred))


class ComponentMetrics:
    """Metrics for component detection."""
    
    @staticmethod
    def iou(
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """
        Compute Intersection over Union for two boxes.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
        
        Returns:
            IoU score (0-1)
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def greedy_match_iou(
        gt_boxes: List[Dict],
        pred_boxes: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Greedy matching of predicted to ground truth boxes by IoU.
        
        Args:
            gt_boxes: Ground truth boxes
            pred_boxes: Predicted boxes
            iou_threshold: Minimum IoU for a match
        
        Returns:
            (matched_gt_indices, matched_pred_indices, matched_ious)
        """
        matched_gt = []
        matched_pred = []
        matched_ious = []
        
        used_gt = set()
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_gt_idx = -1
            best_iou = 0.0
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in used_gt:
                    continue
                
                iou = ComponentMetrics.iou(
                    (gt_box['x1'], gt_box['y1'], gt_box['x2'], gt_box['y2']),
                    (pred_box['x1'], pred_box['y1'], pred_box['x2'], pred_box['y2']),
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                used_gt.add(best_gt_idx)
                matched_gt.append(best_gt_idx)
                matched_pred.append(pred_idx)
                matched_ious.append(best_iou)
        
        return matched_gt, matched_pred, matched_ious


class ConfusionMatrix:
    """Confusion matrix computation."""
    
    @staticmethod
    def compute(
        gt_labels: List[str],
        pred_labels: List[str],
        label_order: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute confusion matrix.
        
        Args:
            gt_labels: Ground truth labels
            pred_labels: Predicted labels
            label_order: Order of labels
        
        Returns:
            DataFrame with confusion matrix
        """
        if not label_order:
            label_order = sorted(set(gt_labels) | set(pred_labels))
        
        cm = pd.DataFrame(
            0,
            index=label_order,
            columns=label_order,
        )
        
        for gt, pred in zip(gt_labels, pred_labels):
            if gt in cm.index and pred in cm.columns:
                cm.loc[gt, pred] += 1
        
        return cm
    
    @staticmethod
    def precision_recall_f1(cm: pd.DataFrame, label: str) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 for a class.
        
        Args:
            cm: Confusion matrix
            label: Class label
        
        Returns:
            (precision, recall, f1)
        """
        tp = cm.loc[label, label]
        fp = cm[label].sum() - tp
        fn = cm.loc[label].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return float(precision), float(recall), float(f1)
    
    @staticmethod
    def macro_f1(cm: pd.DataFrame) -> float:
        """Macro-averaged F1."""
        f1_scores = []
        for label in cm.index:
            _, _, f1 = ConfusionMatrix.precision_recall_f1(cm, label)
            f1_scores.append(f1)
        return np.mean(f1_scores) if f1_scores else 0.0
