# ML Assignment 2 — Bank Marketing (additional) — Streamlit App

## Problem Statement
Build and evaluate multiple classification models to predict whether a customer will subscribe to a term deposit, and deploy an interactive Streamlit app that demonstrates the models and evaluation results.

## Dataset Description
- **Name:** Bank Marketing (additional)
- **File:** `bank-additional-full.csv` (semicolon-delimited)
- **Task:** Binary classification (`y`: yes/no)
- **Instances:** ~41,188; **Features:** ~20 input attributes + target `y`
- **Source:** UCI Machine Learning Repository  
  (See the “Bank Marketing (additional)” dataset page for the CSV and description.)  
  *Reference:* UCI ML Repository — Bank Marketing. [2](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Models Used (all trained on the same dataset)
1. Logistic Regression
2. Decision Tree
3. k‑Nearest Neighbors
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Evaluation Metrics
We report **Accuracy, AUC, Precision, Recall, F1, MCC** for each model.  
(MCC is computed using `sklearn.metrics.matthews_corrcoef`.) [4](https://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/024.pdf)

### Comparison Table (Test Set)
> Fill this table using the app or `python train.py` (generates `model_comparison.csv`).

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|---------:|------:|----------:|------:|------:|------:|
| Logistic Regression |          |       |           |       |       |       |
| Decision Tree       |          |       |           |       |       |       |
| kNN                 |          |       |           |       |       |       |
| Naive Bayes         |          |       |           |       |       |       |
| Random Forest       |          |       |           |       |       |       |
| XGBoost             |          |       |           |       |       |       |

### Observations about Model Performance
- (Add 3–5 bullet points comparing F1/AUC/MCC across models, note bias–variance, etc.)

## Project Structure
