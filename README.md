# ML Assignment 2 — Classification on WDBC (Streamlit + Deployment)

## Problem Statement
Build and evaluate multiple classification models on a public dataset, then deploy an interactive Streamlit app showcasing results.

## Dataset Description
- **Name:** UCI Breast Cancer Wisconsin (Diagnostic)
- **Type:** Binary classification (malignant vs. benign)
- **Instances:** 569
- **Features:** 30 real-valued features
- **Source:** UCI ML Repository; also available via `sklearn.datasets.load_breast_cancer`.
- **Why this dataset?** Meets the assignment constraints (≥12 features, ≥500 instances) and is widely used in ML education.

## Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1, MCC.

### Comparison Table (Test Set)
> 

| ML Model Name          | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|------------------------|---------:|------:|----------:|------:|------:|------:|
| Logistic Regression    |          |       |           |       |       |       |
| Decision Tree          |          |       |           |       |       |       |
| kNN                    |          |       |           |       |       |       |
| Naive Bayes (Gaussian) |          |       |           |       |       |       |
| Random Forest          |          |       |           |       |       |       |
| XGBoost               |          |       |           |       |       |       |

### Observations



## Project Structure