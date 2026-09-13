# Automatic Understanding of Scientific Charts and Diagrams

[![Tests](https://github.com/bmajosek/Automatic-Understanding-of-Scientific-Charts-and-Diagrams/actions/workflows/tests.yml/badge.svg)](https://github.com/bmajosek/Automatic-Understanding-of-Scientific-Charts-and-Diagrams/actions/workflows/tests.yml)

This repository contains the reproducible evaluation and reporting code for the
master's thesis *Automatic Understanding of Scientific Charts and Diagrams*. It
connects final predictions with observable evidence from OCR, table
reconstruction, numerical reasoning, and claim verification.

## Thesis information

- **Author:** Bartosz Maj
- **Programme:** Data Science, Master's degree
- **University:** Warsaw University of Technology (Politechnika Warszawska)
- **Supervisor:** Anna Wróblewska, PhD (dr inż. Anna Wróblewska)

## Main results

The study compares OCR-free image models with explicit OCR, table, and
rule-based pipelines on fixed ChartQA cohorts. In the balanced 1,000-question
local cohort, MatCha ChartQA achieved 63.1% accuracy and Pix2Struct achieved
54.4%. The DePlot table audit reached 74.4% end-to-end tolerant cell accuracy.

These values describe the documented cohorts and configurations. They are not
claims about a new benchmark-wide state of the art.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/` | Evaluation, metrics, plotting, configuration, dataset conversion, and pipeline code |
| `scripts/` | Data preparation, inference, validation, audit, reporting, and security entry points |
| `config/` | Dataset, model, prompt, and evaluation configuration |
| `data/*.csv` | Small schema examples, not the complete experimental datasets |
| `data/processed_review/review_dataset_manifest.json` | Provenance record for the reviewed evaluation cohort |
| `tests/` | Unit and regression tests for the evaluation workflow |

Raw chart images, licensed datasets, model weights, API credentials, complete
prediction files, and large generated result directories are intentionally not
versioned.

## Quick start

Python 3.10 or newer is required. The commands below install the analysis and
test dependencies only. They do not download models or call external APIs.

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

## Safe credential configuration

Copy the example file and add credentials only to the untracked `.env` file:

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

Keep `GEMINI_API_KEY` and `HF_TOKEN` empty in `.env.example`. Never place a
credential in source code, command history, tracked configuration, reports, or
saved model output. If a credential enters a commit, revoke it first and then
remove it from every reachable Git ref.

The project includes a redacting scanner. It reports only the path, location,
and rule name, never the matched value:

```bash
python scripts/check_secrets.py --history --include-untracked
```

GitHub Actions runs the same full-history check on every push and pull request.

## Reproduce analysis from saved artefacts

The recommended review workflow uses saved predictions and does not execute OCR,
local neural models, or Gemini. Place private inputs in the ignored directories
documented by the scripts, then validate the selected experiment:

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

Run any command with `--help` to see its complete interface. Missing or failed
rows remain explicit instead of being silently scored as incorrect.

## Optional dataset and model execution

Dataset acquisition helpers are available through the optional dependency set:

```bash
python -m pip install -e ".[datasets]"
python -m src download-data --dataset chartqa
python -m src prepare-data --dataset chartqa
```

Full inference needs the heavier packages in `requirements_pipelines.txt`. That
file includes a PyTorch build for NVIDIA CUDA 12.6, so CPU, macOS, and other CUDA
environments require a platform-appropriate PyTorch installation.

Inspect a command before starting a potentially long or paid run:

```bash
python scripts/run_experiments.py --help
python scripts/run_gemini_qa_500.py --help
```

Inference is resumable and records failures for later audit.

## Quality checks

```bash
python scripts/check_secrets.py --history --include-untracked
python -m ruff check src scripts tests
python -m pytest -q
```

The automated workflow runs these checks on Python 3.10 and 3.11. It does not
download research datasets or execute a model.

## Reproducibility boundaries

- The pipeline records random seeds, denominators, chart-cluster bootstrap
  grouping, and output manifests.
- Publication figures come from validated saved inputs, not live API calls.
- The public repository provides code and small schemas, but not every
  third-party input required to reproduce the thesis from a fresh clone.
- Dataset and model licences remain the responsibility of their owners.

## Citation

If you use this code, cite the accompanying thesis and [`CITATION.cff`](CITATION.cff).
GitHub also exposes this metadata through its **Cite this repository** action.

## License

No software licence is asserted. Public visibility supports examination and
reproducibility, but does not grant permission to redistribute or reuse the code
or associated data. Contact the author before reuse.
