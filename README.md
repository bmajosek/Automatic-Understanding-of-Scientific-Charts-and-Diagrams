# Automatic Understanding of Scientific Charts and Diagrams

This repository contains the reproducible analysis and reporting code for the master's thesis *Automatic Understanding of Scientific Charts and Diagrams*. It turns validated prediction files into the tables, statistical summaries, audits, and figures used in the manuscript.

## What is included

- deterministic evaluation, bootstrap intervals, paired comparisons, and provenance checks;
- data-preparation and inference entry points for local models and Gemini runs;
- report-generation scripts and publication figures;
- small schema/example CSV files and the review-dataset manifest.

Raw images, licensed datasets, model weights, API credentials, predictions, and large generated result directories are intentionally not versioned. Supply them locally through the paths documented in `config/` and the scripts. Copy `.env.example` to `.env` and add credentials only on your machine.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
```

The optional `datasets` extra installs packages needed for downloading external datasets. The core package is sufficient for analysis of already-saved CSV artefacts.

## Reproduce the analysis

The normal workflow is: prepare the fixed evaluation material, run inference, validate output completeness and provenance, then generate reports and figures from saved artefacts. The main reusable entry points are in `scripts/`; in particular, `generate_experiment_report.py`, `run_review_audit.py`, and `run_gemini_qa_500.py` support resumable, report-only workflows. No model execution is required to inspect or regenerate existing analysis outputs.

Run the automated checks with:

```bash
ruff check src scripts tests
pytest -q
```

## Reproducibility notes

Seeds, evaluation denominators, cluster-bootstrap grouping, and model-output manifests are recorded by the analysis pipeline. Generated artefacts are derived from validated inputs rather than live API calls. Gemini jobs are resumable and preserve failed or missing rows for audit; never commit their credentials or raw responses.

## Thesis and citation

The manuscript is maintained separately from this code repository. The thesis appendix links to this repository and maps its workflow stages to the scripts above. If you use the code, please cite the thesis (see `CITATION.cff`).

## License

No license is asserted in this repository. Contact the author before redistributing or reusing the code or associated data.
