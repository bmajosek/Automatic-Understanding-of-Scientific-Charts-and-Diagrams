"""Generate mixed-sample thesis figures and CSVs from a resumable experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_analysis import build_experiment_analysis, save_analysis_csvs  # noqa: E402
from experiment_plots import generate_experiment_figures  # noqa: E402


RESEARCH_GAPS = """# Research extensions supported by the experiment design

## Highest-value additions

1. **Real scientific-chart generalization.** ChartQA is useful for controlled
   comparison, but the thesis title concerns scientific charts and diagrams.
   Add CharXiv or SCI-CQA, and report descriptive versus reasoning performance.
2. **In-domain versus transfer learning.** Compare the ChartQA-finetuned MatCha
   checkpoint with the PlotQA-finetuned MatCha checkpoint on the same examples.
   This isolates task-domain transfer while keeping the architecture family
   similar.
3. **Error propagation through explicit pipelines.** Relate DePlot
   table-extraction correctness to downstream QA correctness per chart. This
   distinguishes perception failures from reasoning failures. The review
   dataset builder imports ChartQA's distributed reference tables; after DePlot
   inference, `run_review_audit.py` reports key coverage, cell-value accuracy,
   and the same reasoner on reference versus predicted tables.
4. **Explainable verification.** Add ChartCheck-style explanations and score
   both the verdict and whether the cited chart evidence is correct.
5. **Unanswerable and hallucination robustness.** Add absent-information and
   contradictory questions, then report refusal accuracy separately from normal
   QA accuracy. For summaries, evaluate factual support in addition to lexical
   overlap.
6. **Robustness perturbations.** Re-evaluate a fixed subset after changes to
   resolution, font size, color palette, legend position, and JPEG compression.
7. **Scientific context and multi-panel reasoning.** Evaluate whether captions
   or surrounding paper text help, and whether models can connect evidence
   across multiple subplots.
8. **Efficiency.** Record inference time, peak GPU memory, API calls, and
   estimated cost alongside accuracy and coverage.

## Recommended figures once the required data exist

- **Error-propagation cascade:** table-cell correctness → answer validity →
  answer correctness → verification correctness. Requires populated
  `tables_gt.csv` and chart-level joins.
- **Robustness degradation curves:** accuracy change versus image resolution,
  blur, compression, font scaling, color changes and legend displacement.
- **Cross-dataset generalization small multiples:** the same models and metrics
  on ChartQA, CharXiv/SCI-CQA, ChartQAPro and ChartCheck.
- **Chart-attribute difficulty heatmap:** performance by chart type, axis scale,
  legend presence, number of series and multi-panel structure.
- **Summary factuality–coverage plot:** factual support on one axis and
  information coverage on the other; lexical overlap should remain a secondary
  metric.
- **Context ablation chart:** image only versus image + caption versus image +
  surrounding paper text, particularly on scientific and multi-panel figures.
- **Efficiency Pareto frontier:** accuracy against latency, GPU memory, API
  calls or monetary cost.
- **Risk–coverage curve:** only after models provide a meaningful confidence
  signal or abstention rule. Do not infer calibration from API success rates.

## Primary literature motivating these analyses

- CharXiv (realistic charts from scientific papers; descriptive and reasoning
  questions): https://arxiv.org/abs/2406.18521
- SCI-CQA (scientific charts, diagrams, flowcharts and document context):
  https://arxiv.org/abs/2412.12150
- ChartCheck (explainable fact-checking, reasoning types and visual attributes):
  https://arxiv.org/abs/2311.07453
- SciGraphQA (scientific figures, paper context and multi-turn QA):
  https://arxiv.org/abs/2308.03349
- ChartX (18 chart types, seven tasks and 22 disciplinary topics):
  https://arxiv.org/abs/2402.12185
- ChartQAPro (diversity, unanswerable, conversational and hypothetical QA):
  https://arxiv.org/abs/2504.05506
- ChartHal (fine-grained chart hallucination):
  https://arxiv.org/abs/2509.17481
- Chart summarization hallucinations:
  https://arxiv.org/abs/2308.00399
- DePlot (plot-to-table decomposition):
  https://arxiv.org/abs/2212.10505
"""

THESIS_FIGURE_GUIDE = """# Thesis figure guide

All visible performance values are percentages. Exact sample counts and
denominators remain available in the adjacent `csv/` directory.

## Core results chapter

1. **Direct alignment audit figures** — use OCR label coverage, reference-cell
   DePlot accuracy, the same reasoner on reference versus predicted tables, and
   balanced verification. The former alignment-stage proxy figure is
   intentionally suppressed because overlapping final-QA subsets are not
   direct measurements of intermediate alignment.
2. **`fig_current_qa_accuracy_ci`** — primary QA comparison with uncertainty.
   Use this for model ranking, while stating that Gemini confidence intervals
   are wider because only a pilot subset has been evaluated.
3. **`fig_qa_evaluation_status`** — separates experiment completion, execution
   success and answer accuracy. This is preferable to calling quota-limited
   results “API reliability”.
4. **`fig_current_qa_by_operation`** — shows which reasoning operations are
   difficult. It supports analysis of retrieval, arithmetic, comparison, ratio
   and trend errors.
5. **`fig_current_verification_performance`** — reports accuracy and macro-F1
   together. Low macro-F1 despite high accuracy reveals label imbalance or
   prediction collapse.

## Experimental validity and limitations

6. **`fig_current_coverage_matrix`** — makes incomplete Gemini runs explicit.
7. **`fig_current_qa_outcomes`** — decomposes the target into correct,
   incorrect, failed and not-yet-evaluated shares.
8. **`fig_sample_size_precision`** — explains why a 100-sample pilot has wider
   statistical uncertainty than a 1,000-sample run.
9. **`fig_dataset_scope_gap`** — demonstrates that current empirical evidence
   is entirely from ChartQA.

## Methodology and supplementary material

10. **`fig_model_component_matrix`** — maps each method to OCR, structure,
    table extraction, symbolic reasoning, vision-language and cloud components.
11. **`fig_current_summarization_quality`** — preliminary lexical-overlap
    result only; it must not be presented as a complete factuality evaluation.

Use the review-audit table-extraction figure only after DePlot predictions exist
for the same chart cohort as `tables_gt.csv`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create thesis-ready plots from a checkpointed experiment."
    )
    parser.add_argument(
        "--experiment-dir",
        default="results/experiments/all_tasks_1000",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--gemini-qa-limit",
        type=int,
        default=500,
        help="Planned common QA cohort for Gemini-dependent approaches.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = ROOT / experiment_dir
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tolerance = float(config.get("evaluation", {}).get("numerical_tolerance", 0.05))

    predictions_dir = experiment_dir / "predictions"
    analysis_dir = experiment_dir / "analysis"
    csv_dir = analysis_dir / "csv"
    figures_dir = analysis_dir / "figures"
    analysis = build_experiment_analysis(
        data_dir=data_dir,
        predictions_dir=predictions_dir,
        split=args.split,
        limit=args.limit,
        numerical_tolerance=tolerance,
        active_datasets=config.get("datasets", {}).get("active", []),
        gemini_qa_limit=args.gemini_qa_limit or None,
    )
    csv_paths = save_analysis_csvs(analysis, csv_dir)
    figure_paths = generate_experiment_figures(analysis, figures_dir)
    gaps_path = analysis_dir / "RESEARCH_GAPS.md"
    gaps_path.write_text(RESEARCH_GAPS, encoding="utf-8")
    guide_path = analysis_dir / "THESIS_FIGURE_GUIDE.md"
    guide_path.write_text(THESIS_FIGURE_GUIDE, encoding="utf-8")
    manifest = {
        "experiment_dir": str(experiment_dir.resolve()),
        "predictions_dir": str(predictions_dir.resolve()),
        "target_limit": args.limit,
        "gemini_qa_target_limit": args.gemini_qa_limit,
        "split": args.split,
        "csv_files": [str(path.resolve()) for path in csv_paths],
        "figure_files": [str(path.resolve()) for path in figure_paths],
        "research_gaps": str(gaps_path.resolve()),
        "thesis_figure_guide": str(guide_path.resolve()),
    }
    manifest_path = analysis_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Analysis: {analysis_dir.resolve()}")
    print(f"CSV files: {len(csv_paths)}")
    print(f"Figures: {len(figure_paths)} (PNG and PDF)")
    print(f"Manifest: {manifest_path.resolve()}")
    qa = analysis["qa_model_summary"]
    if not qa.empty:
        print("\nQA summary:")
        print(qa[[
            "model_label", "valid_samples", "coverage", "accuracy",
            "ci95_low", "ci95_high",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
