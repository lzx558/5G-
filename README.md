# 5G User Prediction

A machine learning pipeline that predicts whether a telecom user is a **5G subscriber** based on profile and usage features. This project implements three classifiers, evaluates them with **AUC (Area Under the ROC Curve)**, and exports metrics plus visualization artifacts for the assignment report.

## Problem Description

Given user-side information (billing, data usage, activity, plan type, region, etc.), the task is to build a binary classifier:

| Item     | Detail                                                       |
| -------- | ------------------------------------------------------------ |
| Task     | Binary classification (5G user vs. non-5G user)              |
| Features | 20 categorical (`cat_0`–`cat_19`) + 38 numerical (`num_0`–`num_37`) |
| Label    | `target` — `1` = 5G user, `0` = non-5G user                  |
| Metric   | **AUC** — higher is better; robust to severe class imbalance |

## Dataset

- **File:** `train.csv`
- **Samples:** 800,000
- **Columns:** 60 (`id`, 20 categorical features, 38 numerical features, `target`)
- **Class imbalance:** ~1.33% positive (5G users)

Place `train.csv` in the project root before running the script.

## Models

| Model                   | Role                                          | Imbalance Handling        |
| ----------------------- | --------------------------------------------- | ------------------------- |
| **LightGBM**            | Gradient boosting; native categorical support | `scale_pos_weight`        |
| **Logistic Regression** | Linear baseline; fast and interpretable       | `class_weight='balanced'` |
| **Random Forest**       | Ensemble of decision trees                    | `class_weight='balanced'` |

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline (EDA, training, evaluation, and figure export):

```bash
python main.py
```

### What the script does

1. Loads and splits data (80% train / 20% validation, stratified)
2. Generates exploratory plots under `outputs/figures/`
3. Trains LightGBM, Logistic Regression, and Random Forest
4. Computes validation AUC for each model
5. Saves metrics, predictions, and comparison plots

## Outputs

After a successful run, results are written to `outputs/`:

| Path                                          | Description                                     |
| --------------------------------------------- | ----------------------------------------------- |
| `outputs/metrics.json`                        | Validation AUC per model and best-model summary |
| `outputs/validation_predictions.csv`          | Ground-truth labels and predicted probabilities |
| `outputs/figures/target_distribution.png`     | Class distribution                              |
| `outputs/figures/num_feature_correlation.png` | Top numerical features correlated with `target` |
| `outputs/figures/cat_feature_target_rate.png` | 5G rate by categorical feature value            |
| `outputs/figures/roc_curves.png`              | ROC curves for all models                       |
| `outputs/figures/feature_importance.png`      | LightGBM top-20 feature importances             |

## Evaluation Metric

AUC is computed with scikit-learn:

```python
from sklearn.metrics import roc_auc_score

y_pred = model.predict_proba(X_val)[:, 1]  # predicted probability of class 1
auc = roc_auc_score(y_true, y_pred)
```

Example validation results (reference):

| Model               | Validation AUC |
| ------------------- | -------------- |
| Random Forest       | 0.9060         |
| LightGBM            | 0.9016         |
| Logistic Regression | 0.8407         |

## Project Structure

```
.
├── main.py              # Training and evaluation pipeline
├── train.csv            # Training data (not included in repo)
├── requirements.txt     # Python dependencies
├── README.md            # This file (English)
├── 作业报告.md           # Assignment report (Chinese, no source code)
└── outputs/             # Generated after running main.py
    ├── metrics.json
    ├── validation_predictions.csv
    └── figures/
```

## Report

The written analysis (task background, EDA, model rationale, comparison, and improvement ideas) is in **`作业报告.md`** (Chinese). It contains pseudocode only—no experiment source code—as required by the assignment.

## Notes

- Categorical columns are label-encoded for sklearn models and passed as `category` dtype to LightGBM.
- Numerical columns are standardized for Logistic Regression and Random Forest.
- Plot labels use English; chart titles support Chinese fonts on Windows when available.

## License

Academic / coursework use.
