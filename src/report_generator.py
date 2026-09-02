"""
Report generation for thesis Results chapter.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def _safe_str(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


class ReportGenerator:
    """Generate Markdown reports."""
    
    @staticmethod
    def generate_data_description(
        charts: pd.DataFrame,
        questions: pd.DataFrame,
        claims: pd.DataFrame,
        components: pd.DataFrame,
    ) -> str:
        """Generate data description section."""
        report = "## Data Description\n\n"
        
        if not charts.empty:
            num_charts = len(charts)
            num_unique_types = charts['chart_type'].nunique()
            num_sources = charts['source'].nunique()
            
            report += "### Chart Metadata\n\n"
            report += f"- Total charts: {num_charts}\n"
            report += f"- Unique chart types: {num_unique_types}\n"
            report += f"- Unique sources: {num_sources}\n"
            
            # Chart type distribution
            report += "\n#### Chart Type Distribution\n\n"
            type_dist = charts['chart_type'].value_counts()
            for chart_type, count in type_dist.items():
                report += f"- {chart_type}: {count}\n"
            
            # Source distribution
            report += "\n#### Chart Distribution by Source\n\n"
            source_dist = charts['source'].value_counts()
            for source, count in source_dist.items():
                report += f"- {source}: {count}\n"
            
            # Split distribution
            if 'split' in charts.columns:
                report += "\n#### Split Distribution\n\n"
                split_dist = charts['split'].value_counts()
                for split, count in split_dist.items():
                    report += f"- {split}: {count}\n"
        
        if not questions.empty:
            report += "\n### Questions\n\n"
            num_questions = len(questions)
            report += f"- Total questions: {num_questions}\n"
            
            if 'task' in questions.columns:
                task_dist = questions['task'].value_counts()
                report += f"- Tasks: {task_dist.to_dict()}\n"
            
            if 'answer_type' in questions.columns:
                answer_type_dist = questions['answer_type'].value_counts()
                report += f"- Answer types: {answer_type_dist.to_dict()}\n"
        
        if not claims.empty:
            report += "\n### Claims\n\n"
            num_claims = len(claims)
            report += f"- Total claims: {num_claims}\n"
            
            if 'label' in claims.columns:
                label_dist = claims['label'].value_counts()
                report += "- Label distribution:\n"
                for label, count in label_dist.items():
                    report += f"  - {label}: {count}\n"
        
        if not components.empty:
            report += "\n### Components\n\n"
            num_components = len(components)
            report += f"- Total component annotations: {num_components}\n"
            
            if 'component_type' in components.columns:
                type_dist = components['component_type'].value_counts()
                report += f"- Component types: {type_dist.to_dict()}\n"
        
        return report
    
    @staticmethod
    def generate_model_setup(
        models: pd.DataFrame,
        evaluated_model_names: Optional[List[str]] = None,
    ) -> str:
        """Generate model setup section."""
        report = "## Evaluated Models and Experimental Setup\n\n"
        evaluated = set(evaluated_model_names or [])

        explicit = {
            "classical_cv_ocr_pipeline", "chartocr_reasoning_pipeline",
            "ocr_gemini_reasoning_pipeline", "deplot_table_gemini_pipeline",
            "pix2struct_ocr_free_pipeline", "matcha_chartqa_ocr_free_pipeline",
            "matcha_plotqa_transfer_pipeline", "table_symbolic_reasoner_pipeline",
        }
        implicit = {"gemini_end_to_end"}

        report += "### Method families\n\n"
        report += "- **Explicit pipelines**: OCR, chart structure, DePlot table, or symbolic table reasoning.\n"
        report += "- **Implicit (end-to-end)**: `gemini_end_to_end` — image + question only (single Gemini model).\n"
        report += "- **Baselines**: `constant_answer_baseline`, `train_prior_baseline` (no image).\n"
        report += "- **Debug only**: `oracle_upper_bound` copies gold answers; do not report as model performance.\n\n"

        if models.empty:
            report += "No model metadata in models.csv.\n"
            if evaluated:
                report += f"\nModels with prediction files evaluated: {', '.join(sorted(evaluated))}\n"
            return report

        report += "### Model Summary\n\n"
        
        for _, row in models.iterrows():
            model_name = row.get('model_name', 'Unknown')
            family = row.get('model_family', 'Unknown')
            approach = row.get('approach', 'Unknown')
            notes = _safe_str(row.get('notes', ''))
            status = "evaluated" if model_name in evaluated else "not evaluated (no predictions or zero matches)"
            if "PLANNED" in notes.upper() or "planned" in notes.lower():
                status = "planned — not evaluated"
            if model_name in explicit:
                family_tag = "explicit pipeline"
            elif model_name in implicit:
                family_tag = "implicit Gemini"
            elif model_name == "oracle_upper_bound":
                family_tag = "debug oracle"
            else:
                family_tag = str(family)
            
            report += f"**{model_name}** ({family_tag}; {status})\n\n"
            report += f"- Family: {family}\n"
            report += f"- Approach: {approach}\n"
            
            if pd.notna(row.get('temperature')):
                report += f"- Temperature: {row.get('temperature')}\n"
            
            if pd.notna(row.get('max_tokens')):
                report += f"- Max tokens: {row.get('max_tokens')}\n"
            
            if pd.notna(row.get('prompt_version')):
                report += f"- Prompt version: {row.get('prompt_version')}\n"
            
            if pd.notna(row.get('input_type')):
                report += f"- Input type: {row.get('input_type')}\n"
            
            if pd.notna(row.get('notes')):
                report += f"- Notes: {row.get('notes')}\n"
            
            report += "\n"
        
        return report
    
    @staticmethod
    def _format_errors(error_dist: Dict) -> str:
        if not error_dist:
            return "ok"
        top = sorted(error_dist.items(), key=lambda x: -x[1])[:2]
        return "; ".join(f"{k}: {v}" for k, v in top)

    @staticmethod
    def generate_qa_results(qa_results: List[Dict]) -> str:
        """Generate QA results section."""
        report = "## Chart Question Answering Results\n\n"
        
        qa_results = [
            r for r in qa_results
            if r.get("evaluated_questions", 0) > 0
            and r.get("model_name") != "oracle_upper_bound"
        ]
        if not qa_results:
            report += "No QA results available (no models with matching predictions).\n"
            report += "Run pipelines under `predictions/<model_name>/qa_pred.csv` then re-evaluate.\n"
            return report

        pilot = [r for r in qa_results if r.get("is_pilot_run")]
        full = [r for r in qa_results if not r.get("is_pilot_run")]

        if pilot:
            report += (
                "### Pipeline pilot runs (subset of test questions)\n\n"
                "These runs use `--limit` on the test split. Accuracy is computed only "
                "on valid predictions (n in table). Runtime/API failures, including "
                "Gemini quota exhaustion, are excluded from accuracy and reported "
                "separately as coverage failures.\n\n"
            )
            report += "| Model | Accuracy | n | Coverage of GT | Top errors |\n"
            report += "|-------|----------|---|----------------|------------|\n"
            for result in sorted(pilot, key=lambda x: -x.get("accuracy", 0)):
                report += (
                    f"| {result.get('model_name')} | {result.get('accuracy', 0):.3f} | "
                    f"{result.get('evaluated_questions', 0)} | "
                    f"{result.get('coverage_of_gt', 0)*100:.2f}% | "
                    f"{ReportGenerator._format_errors(result.get('error_type_distribution', {}))} |\n"
                )
            report += "\n"

        if full:
            report += "### Full-coverage runs (baselines on entire question set)\n\n"
            report += "| Model | Accuracy | n | Notes |\n"
            report += "|-------|----------|---|-------|\n"
            for result in sorted(full, key=lambda x: -x.get("accuracy", 0)):
                report += (
                    f"| {result.get('model_name')} | {result.get('accuracy', 0):.3f} | "
                    f"{result.get('evaluated_questions', 0)} | non-visual baseline |\n"
                )
            report += "\n"

        report += (
            "> **Note:** Do not compare pilot n≈10 pipeline accuracy directly to "
            "full-set baseline accuracy. Extend `--limit` or remove it for full test-set "
            "pipeline evaluation.\n\n"
        )
        
        if pilot:
            report += "### Accuracy by Task (pilot pipelines only)\n\n"
            for result in pilot:
                by_task = result.get('accuracy_by_task', {})
                if by_task:
                    report += f"**{result.get('model_name')}** (n={result.get('evaluated_questions', 0)}):\n"
                    for task, accuracy in sorted(by_task.items()):
                        report += f"- {task}: {accuracy:.3f}\n"
                    report += "\n"
        
        return report
    
    @staticmethod
    def generate_table_extraction_results(table_results: List[Dict]) -> str:
        """Generate table extraction results section."""
        report = "## Chart-to-Table Extraction Results\n\n"
        
        if not table_results:
            report += "No table extraction results available.\n"
            return report
        
        report += "### Overall Performance\n\n"
        
        report += "| Model | Charts | Cells | MAE | MRE | Accuracy |\n"
        report += "|-------|--------|-------|-----|-----|----------|\n"
        
        for result in table_results:
            model = result.get('model_name', 'Unknown')
            charts = result.get('num_evaluated_charts', 0)
            cells = result.get('num_evaluated_cells', 0)
            mae = result.get('mae', np.nan)
            mre = result.get('mre', np.nan)
            accuracy = result.get('accuracy_with_tolerance', np.nan)
            
            mae_str = f"{mae:.3f}" if not np.isnan(mae) else '--'
            mre_str = f"{mre:.3f}" if not np.isnan(mre) else '--'
            acc_str = f"{accuracy:.3f}" if not np.isnan(accuracy) else '--'
            
            report += f"| {model} | {charts} | {cells} | {mae_str} | {mre_str} | {acc_str} |\n"
        
        report += "\n"
        
        return report
    
    @staticmethod
    def generate_verification_results(verification_results: List[Dict]) -> str:
        """Generate verification results section."""
        report = "## Factual Verification and Explanation Grounding\n\n"
        
        if not verification_results:
            report += "No verification results available.\n"
            return report

        warnings = sorted({
            r.get("single_class_warning", "") for r in verification_results
            if r.get("single_class_warning")
        })
        for warning in warnings:
            report += f"> **Validity warning:** {warning}\n\n"
        
        report += "### Overall Performance\n\n"
        
        report += "| Model | Claims | Accuracy | Macro F1 | Avg Grounding |\n"
        report += "|-------|--------|----------|----------|---------------|\n"
        
        for result in verification_results:
            model = result.get('model_name', 'Unknown')
            claims = result.get('evaluated_claims', 0)
            accuracy = result.get('accuracy', np.nan)
            macro_f1 = result.get('macro_f1', np.nan)
            grounding = result.get('average_grounding', np.nan)
            
            acc_str = f"{accuracy:.3f}" if not np.isnan(accuracy) else '--'
            f1_str = f"{macro_f1:.3f}" if not np.isnan(macro_f1) else '--'
            gnd_str = f"{grounding:.3f}" if not np.isnan(grounding) else '--'
            
            report += f"| {model} | {claims} | {acc_str} | {f1_str} | {gnd_str} |\n"
        
        report += "\n"
        
        # Per-class metrics
        report += "### Per-Class Metrics\n\n"
        for result in verification_results:
            model = result.get('model_name', 'Unknown')
            per_class = result.get('per_class_metrics', {})
            
            if per_class:
                report += f"**{model}**:\n\n"
                report += "| Label | Precision | Recall | F1 |\n"
                report += "|-------|-----------|--------|----|\n"
                
                for label, metrics in per_class.items():
                    precision = metrics.get('precision', 0.0)
                    recall = metrics.get('recall', 0.0)
                    f1 = metrics.get('f1', 0.0)
                    report += f"| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} |\n"
                
                report += "\n"
        
        return report
    
    @staticmethod
    def generate_interpretation(
        qa_results: List[Dict],
        table_results: List[Dict],
        verification_results: List[Dict],
    ) -> str:
        """Generate interpretation section."""
        report = "## Interpretation of Results\n\n"
        qa_scored = [
            r for r in qa_results
            if r.get("evaluated_questions", 0) > 0 and r.get("model_name") != "oracle_upper_bound"
        ]
        pilot = [r for r in qa_scored if r.get("is_pilot_run")]
        full = [r for r in qa_scored if not r.get("is_pilot_run")]

        report += "### Performance overview\n\n"
        if pilot:
            best = max(pilot, key=lambda x: x.get("accuracy", 0))
            report += (
                f"- Best pilot pipeline on the evaluated subset: "
                f"**{best.get('model_name')}** ({best.get('accuracy', 0):.3f}, "
                f"n={best.get('evaluated_questions', 0)}).\n"
            )
            explicit = [r for r in pilot if r.get("model_name") in {
                "classical_cv_ocr_pipeline", "chartocr_reasoning_pipeline",
                "ocr_gemini_reasoning_pipeline", "deplot_table_gemini_pipeline",
                "pix2struct_ocr_free_pipeline", "matcha_chartqa_ocr_free_pipeline",
                "matcha_plotqa_transfer_pipeline", "table_symbolic_reasoner_pipeline",
            }]
            if explicit:
                avg_exp = np.mean([r.get("accuracy", 0) for r in explicit])
                report += f"- Mean explicit-pipeline pilot accuracy: {avg_exp:.3f}.\n"
        if full:
            report += (
                f"- Non-visual baselines on the full question set: "
                f"train-prior {next((r.get('accuracy',0) for r in full if 'train_prior' in r.get('model_name','')), 0):.3f}, "
                f"constant {next((r.get('accuracy',0) for r in full if 'constant' in r.get('model_name','')), 0):.3f}.\n"
            )
        report += "\n"

        report += "### Key observations\n\n"
        report += (
            "- **Explicit vs implicit:** Explicit pipelines expose OCR/table intermediates; "
            "implicit Gemini uses image+question only (`gemini_end_to_end`). Compare them on the "
            "same n by re-running with matching `--limit`.\n"
        )
        if pilot:
            ocr_fail = [r for r in pilot if any(
                "ocr_error" in str(k) or "Tesseract" in str(k)
                for k in r.get("error_type_distribution", {})
            )]
            if ocr_fail:
                report += (
                    "- **OCR failures:** Some pilot runs report Tesseract errors — set "
                    "`TESSDATA_PREFIX` to the `tessdata` folder in `.env` (see README).\n"
                )
            hf_fail = [r for r in pilot if any("hf_error" in str(k) or "RuntimeError" in str(k)
                       for k in r.get("error_type_distribution", {}))]
            if hf_fail:
                report += (
                    "- **Hugging Face models:** Pix2Struct/DePlot require `torch`, `transformers`, "
                    "and optional `HF_TOKEN` for gated weights.\n"
                )
        report += (
            "- **DePlot + Gemini** often leads pilot subsets when table context fits the question type.\n"
            "- **Oracle** is excluded from analysis; it only validates the evaluation harness.\n"
        )
        report += "\n"

        report += "### Limitations\n\n"
        report += (
            "- Pilot `--limit` runs are not full benchmark scores.\n"
            "- Multi-dataset merged evaluation requires running `prepare-data --all` after downloads.\n"
            "- Verification and table metrics need matching prediction files per task.\n"
        )
        report += "\n"
        return report
    
    @staticmethod
    def generate_full_report(
        charts: pd.DataFrame,
        questions: pd.DataFrame,
        claims: pd.DataFrame,
        components: pd.DataFrame,
        models: pd.DataFrame,
        qa_results: List[Dict],
        table_results: List[Dict],
        verification_results: List[Dict],
    ) -> str:
        """Generate complete results chapter draft."""
        report = "# Results\n\n"
        evaluated_names = [r.get("model_name") for r in qa_results if r.get("evaluated_questions", 0) > 0]
        
        report += ReportGenerator.generate_data_description(
            charts, questions, claims, components
        )
        report += "\n"
        
        report += ReportGenerator.generate_model_setup(models, evaluated_names)
        report += "\n"
        
        report += ReportGenerator.generate_table_extraction_results(table_results)
        report += "\n"
        
        report += ReportGenerator.generate_qa_results(qa_results)
        report += "\n"
        
        report += ReportGenerator.generate_verification_results(verification_results)
        report += "\n"
        
        report += ReportGenerator.generate_interpretation(
            qa_results, table_results, verification_results
        )
        
        return report
