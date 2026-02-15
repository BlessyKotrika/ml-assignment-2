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

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
| ------------------- | -------- | ----- | --------- | ------ | ----- | ----- |
| Logistic Regression | 0.854    | 0.911 | 0.422     | 0.821  | 0.558 | 0.519 |
| XGBoost             | 0.902    | 0.924 | 0.566     | 0.536  | 0.550 | 0.496 |
| Decision Tree       | 0.868    | 0.696 | 0.421     | 0.473  | 0.445 | 0.372 |
| Naive Bayes         | 0.781    | 0.799 | 0.293     | 0.679  | 0.410 | 0.340 |
| Random Forest       | 0.902    | 0.921 | 0.640     | 0.286  | 0.395 | 0.384 |
| kNN                 | 0.888    | 0.876 | 0.500     | 0.304  | 0.378 | 0.332 |


### Observations about Model Performance

Best Overall AUC: XGBoost (0.924) slightly outperforms Random Forest and Logistic Regression, indicating strong ranking ability and good class separability.

Best Recall: Logistic Regression (0.821) captures the highest number of actual subscribers, making it suitable when minimizing false negatives is important (e.g., not missing potential customers).

Best Precision: Random Forest (0.640) produces fewer false positives, making it better when marketing cost per contact is high.

Best MCC: Logistic Regression (0.519) performs best in terms of balanced correlation between predicted and true classes, especially important due to class imbalance.

Bias–Variance Insight:

Decision Tree shows lower AUC (0.696), suggesting possible overfitting (high variance).

Naive Bayes has lower accuracy but decent recall, indicating high bias due to strong independence assumptions.

Ensemble models (Random Forest, XGBoost) improve stability and generalization.

