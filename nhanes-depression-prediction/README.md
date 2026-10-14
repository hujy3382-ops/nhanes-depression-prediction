# Depression Risk Prediction from Routine Clinical Data (NHANES 2017–2018)

> **TODO (Day 6):** Replace every `[…]` below with your own results, then delete this line.

## Overview
Depression is common but under-detected in routine care. This project asks whether
**routinely collected demographic, anthropometric and laboratory data** — including the
inflammatory marker hs-CRP — can identify adults with clinically relevant depressive
symptoms (PHQ-9 ≥ 10).

It is built as a **reproducible, modular ML pipeline**: one command downloads the public
data, cleans it, trains and compares models with cross-validation, evaluates them on a
held-out test set, and explains the best model with SHAP.

## Data
- **Source:** [NHANES 2017–2018](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017), CDC (public, no registration)
- **Modules:** DEMO, DPQ (PHQ-9), HSCRP, BMX, CBC, GHB, HDL, TCHOL
- **Population:** adults ≥ 18 with complete PHQ-9 — final N = [ ], prevalence = [ ]%
- **Outcome:** PHQ-9 total ≥ 10
- **Predictors:** [list, or refer to `config.yaml`]

Raw data are not stored in this repo; `src/download.py` fetches them from the CDC.

## Methods
- **Preprocessing:** PHQ-9 refusal/don't-know codes (7/9) set to missing; hs-CRP log-transformed
- **Leakage control:** imputation, scaling and one-hot encoding are fitted inside a scikit-learn `Pipeline`, i.e. on training folds only
- **Models:** Logistic Regression, Random Forest, XGBoost (class weights for imbalance)
- **Validation:** stratified 80/20 train–test split; 5-fold stratified CV on the training set
- **Metrics:** AUROC, AUPRC, sensitivity, specificity, calibration
- **Explainability:** SHAP (TreeExplainer) on XGBoost

## Results
| Model | CV AUROC | CV AUPRC | Test AUROC | Test AUPRC |
|---|---|---|---|---|
| Logistic Regression | [ ] | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] | [ ] |
| XGBoost | [ ] | [ ] | [ ] | [ ] |

<p float="left">
  <img src="figures/roc.png" width="45%"/>
  <img src="figures/shap_summary.png" width="45%"/>
</p>

**Key findings:** [2–3 sentences, e.g. which features mattered most and whether hs-CRP contributed]

## Limitations
- Cross-sectional data: associations, not causal effects
- PHQ-9 is a screening tool, not a clinical diagnosis
- No external validation; survey weights not applied
- [anything else you noticed]

## How to run
```bash
git clone https://github.com/<your-username>/nhanes-depression-prediction.git
cd nhanes-depression-prediction
pip install -r requirements.txt
python main.py          # full pipeline
pytest -v               # unit tests
```

## Project structure
```
├── main.py              # end-to-end pipeline entry point
├── config.yaml          # paths, variables, hyperparameters
├── src/
│   ├── download.py      # fetch NHANES .xpt files
│   ├── preprocess.py    # merge, clean, score PHQ-9
│   ├── train.py         # pipelines, CV, model fitting
│   ├── evaluate.py      # metrics and plots
│   └── explain.py       # SHAP
├── tests/               # pytest unit tests
├── notebooks/           # results walkthrough
├── figures/  results/   # generated outputs
```

## Author
Jiayi Hu — MSc Cognitive Neuroimaging and Data Science, University of Birmingham
