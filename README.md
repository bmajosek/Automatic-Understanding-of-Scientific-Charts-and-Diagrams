# Automatic Understanding of Scientific Charts and Diagrams

[![Tests](https://github.com/bmajosek/Automatic-Understanding-of-Scientific-Charts-and-Diagrams/actions/workflows/tests.yml/badge.svg)](https://github.com/bmajosek/Automatic-Understanding-of-Scientific-Charts-and-Diagrams/actions/workflows/tests.yml)

Reproducibility repository for the master's thesis *Automatic Understanding of
Scientific Charts and Diagrams* by Bartosz Maj, Warsaw University of Technology,
2026.

The project evaluates where chart-reading systems lose the evidence needed to
support a final answer. It connects final predictions with four observable
levels: text recognition, table reconstruction, numerical reasoning, and claim
verification.

## Thesis at a glance

The study compares OCR-free image models with explicit OCR-, table-, and
rule-based pipelines on fixed ChartQA cohorts. In the main 1,000-question QA
cohort, MatCha ChartQA achieved 63.1% accuracy and Pix2Struct achieved 54.4%.
DePlot reached 74.4% end-to-end tolerant cell accuracy in the table audit. The
main methodological contribution is an evaluation protocol that retains
intermediate outputs, making it possible to locate failures that are hidden by
a final answer alone.

These figures describe the fixed cohorts and conditions reported in the thesis;
they are not claims about a new benchmark-wide state of the art.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/` | Evaluation, metrics, plotting, configuration, dataset conversion, and pipeline code |
| `scripts/` | Reproducible command-line entry points for preparation, inference, validation, audits, and reports |
| `config/` | Dataset, model, prompt, and evaluation configuration |
| `data/*.csv` | Small schema examples, not the complete experimental datasets |
| `data/processed_review/review_dataset_manifest.json` | Provenance record for the reviewed evaluation cohort |
| `tests/` | Unit and regression tests for the repaired evaluation workflow |

Raw chart images, licensed datasets, model weights, API credentials, complete
prediction files, and large generated result directories are intentionally not
versioned.

## Quick start

Python 3.10 or newer is required. The commands below install the analysis and
test dependencies only; they do not download models or call external APIs.

### Windows PowerShell

```powershell
git clone https://github.com/bmajosek/Automatic-Understanding-of-Scientific-Charts-and-Diagrams.git
Set-Location Automatic-Understanding-of-Scientific-Charts-and-Diagrams
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src --help
```

### macOS or Linux

```bash
git clone https://github.com/bmajosek/Automatic-Understanding-of-Scientific-Charts-and-Diagrams.git
cd Automatic-Understanding-of-Scientific-Charts-and-Diagrams
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest -q
.venv/bin/python -m src --help
```

List the datasets known to the project with:

```bash
python -m src list-datasets
```

If the virtual environment is not activated, replace `python` in later examples
with `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS and
Linux.

## Reproduce analysis from saved artefacts

This is the recommended route for inspecting an existing run. It does not
execute OCR, local neural models, or Gemini. Place the private/unversioned files
in the following layout:

```text
data/processed/
  charts.csv
  questions.csv
  claims.csv
  tables_gt.csv
results/experiments/all_tasks_1000/
  predictions/
    <model_name>/
      qa_pred.csv
      verification_pred.csv
      table_extraction_pred.csv
      summarization_pred.csv
```

Only the files relevant to a selected task are required. The scripts preserve
missing rows as missing rather than silently treating them as incorrect.

Validate prediction completeness and provenance:

```bash
python scripts/validate_experiment_outputs.py --data-dir data/processed --predictions-dir results/experiments/all_tasks_1000/predictions --task qa --limit 1000
```

Generate the statistical summaries and publication figures:

```bash
python scripts/generate_experiment_report.py --data-dir data/processed --experiment-dir results/experiments/all_tasks_1000
```

Generate the supervisor-review audit from the same saved files:

```bash
python scripts/run_review_audit.py --data-dir data/processed --experiment-dir results/experiments/all_tasks_1000
```

Generated CSV summaries, figures, and manifests are written below the selected
experiment directory. Run any command with `--help` to see all available
options.

## Optional dataset and model execution

Dataset acquisition helpers can be installed with:

```bash
python -m pip install -e ".[datasets]"
python -m src download-data --dataset chartqa
python -m src prepare-data --dataset chartqa
```

Full inference has substantially heavier dependencies. The supplied
`requirements_pipelines.txt` includes a PyTorch build pinned for NVIDIA CUDA
12.6. CPU, macOS, and other CUDA environments require a platform-appropriate
PyTorch installation instead.

For an NVIDIA CUDA 12.6 environment, install the complete inference stack with:

```bash
python -m pip install -r requirements_pipelines.txt
```

Before using Gemini or gated Hugging Face resources, create a local environment
file and fill in only the credentials you need:

```powershell
Copy-Item .env.example .env       # Windows PowerShell
```

```bash
cp .env.example .env             # macOS or Linux
```

The model selected for Gemini runs is configured in `config/config.yaml`.
After installing the inference dependencies, inspect the interface before
starting a potentially long or paid run:

```bash
python scripts/run_experiments.py --help
python scripts/run_gemini_qa_500.py --help
```

Inference is resumable and records failed or missing rows for audit. Never
commit `.env`, raw Gemini responses, licensed datasets, or model checkpoints.

## Quality checks

```bash
ruff check src scripts tests
pytest -q
```

The GitHub Actions workflow runs the same checks on Python 3.10 and 3.11. It
does not download research datasets or execute any model.

## Reproducibility boundaries

- Random seeds, denominators, chart-cluster bootstrap grouping, and output
  manifests are recorded by the analysis pipeline.
- Publication figures are generated from validated, saved inputs rather than
  live API calls.
- The public repository contains the code and small schemas needed to inspect
  the method, but not every third-party input required to recreate the thesis
  results from a fresh clone.
- Dataset and model licences remain the responsibility of their respective
  owners.

## Citation

If you use this code, cite the accompanying thesis and the software metadata in
[`CITATION.cff`](CITATION.cff). GitHub also exposes this file through its
**Cite this repository** action.

## License

No software licence is asserted. The repository is public for examination and
reproducibility; public visibility does not by itself grant permission to
redistribute or reuse the code or associated data. Contact the author before
reuse.
