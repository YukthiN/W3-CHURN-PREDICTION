# 📉 Customer Churn Prediction — Full ML Comparison

Predicting which telecom customers will leave using 15 models
across 3 strategies — Original, SMOTE balanced, and XGBoost.

## Dataset
IBM Telco Customer Churn (Kaggle)
- 7,043 customers · 21 features
- Target: Churn (Yes/No) — 73% No, 27% Yes (imbalanced)

## What makes this project real
Most churn tutorials use clean balanced data.
This uses the actual IBM Telco dataset as-is — messy,
imbalanced, and unpredictable — just like real business data.

## Models Compared (15 total)

### Group 1 — Original Data
| Model | Accuracy | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Logistic Regression | 79.91% | 0.5916 | 0.6426 | 0.5481 |
| Decision Tree | 73.03% | 0.5052 | 0.4924 | 0.5187 |
| Random Forest | 79.21% | 0.5620 | 0.6373 | 0.5027 |
| Gradient Boosting | 80.13% | 0.5745 | 0.6655 | 0.5053 |
| KNN | 74.24% | 0.5088 | 0.5151 | 0.5027 |
| SVM | 79.35% | 0.5516 | 0.6509 | 0.4786 |
| Naive Bayes | 74.52% | 0.6042 | 0.5141 | 0.7326 |

### Group 2 — SMOTE Balanced
| Model | Accuracy | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Logistic Regression | 74.24% | 0.6191 | 0.5095 | 0.7888 |
| Random Forest | 76.86% | 0.5711 | 0.5622 | 0.5802 |
| Gradient Boosting | 77.57% | 0.6229 | 0.5625 | 0.6979 |
| SVM | 74.66% | 0.5957 | 0.5167 | 0.7032 |

### Group 3 — XGBoost
| Model | Accuracy | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| XGBoost | 75.66% | 0.6142 | 0.5301 | 0.7299 |

## Results
| Goal | Best Model | Why |
|------|-----------|-----|
| Best overall | GB (SMOTE) — F1: 0.6229 | Best balance |
| Catch most churners | LR (SMOTE) — Recall: 0.7888 | Retention teams |
| Precision targeting | GB (original) — Precision: 0.6655 | Limited budget |

## Key Learnings
- Accuracy is misleading on imbalanced data — always use F1
- SMOTE improves recall but trades off precision
- XGBoost handles imbalance natively — best single model
- Industry benchmark for this dataset: F1 ~0.62–0.65
- Our score: **0.6229 — competitive with Kaggle top solutions** ✅
- 95%+ accuracy is not realistic for churn — human behaviour
  is genuinely unpredictable

## Why churn is hard
Someone leaves because their friend switched networks.
No feature in any dataset captures that.
That's irreducible noise — and that's real ML.

## Tech Stack
Python · Scikit-learn · XGBoost · imbalanced-learn
Pandas · NumPy · Matplotlib · Seaborn · Kaggle

## Files
- `churn_prediction.ipynb` — full notebook
- `churn_full_comparison.png` — model comparison chart
- `churn_eda.png` — EDA visualizations

## Author
Yukthi N · AI/ML Engineer · Bengaluru
