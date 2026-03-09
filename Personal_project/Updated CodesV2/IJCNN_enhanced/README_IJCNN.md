# IJCNN Reproducible Code (Enhanced)

This folder contains the reproducible implementation for the paper pipeline:

1. **Neural training** using standard optimizers (Adam / SGD / RMSprop / AdaGrad / AdaDelta / NAdam / Lion)
2. **Hyperparameter optimization (HPO)** using evolutionary algorithms (GA, DE, PSO) and Bayesian optimization (Optuna)
3. **Neuro-symbolic safety layer** combining Rules + Fuzzy risk + Bayesian posterior + NN probability

## Key evaluation rule (publication standard)
The MLP is trained with `BCEWithLogitsLoss`, so the model output is a **logit**.
All evaluation must convert logits to probabilities via **sigmoid**.

Additionally, **decision threshold tuning is performed on VALIDATION only** (never on the test set).

This repository centralizes these rules in:
- `src/utils/inference.py` (logits -> sigmoid probabilities)
- `src/models/metrics.py` (threshold selection + metric computation)
- `src/models/evaluate.py` (VAL-tuned threshold, TEST reporting)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run baseline (train + val threshold + test metrics)
```bash
python -c "from src.pipeline import run_pipeline; run_pipeline()"
```

### Run full HPO
```bash
python -c "from src.pipeline import run_all_hpo; run_all_hpo()"
```

### Generate paper tables/figures
```bash
python -c "from src.analysis.generate_paper_outputs import main; main()"
```

## Dataset
The code loads the CDC Diabetes Health Indicators dataset.
- If `diabetes_binary_health_indicators_BRFSS2015.csv` exists locally in the project root, it will be used.
- Otherwise the dataset is fetched via `ucimlrepo` (ID=891).

## Notes for reviewers
- No test leakage: threshold selection uses validation only.
- HPO evaluators use ROC-AUC computed on **probabilities**.
