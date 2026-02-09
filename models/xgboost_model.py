from xgboost import XGBClassifier

def build(n_estimators: int = 300, random_state: int = 42):
    return XGBClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="auto"
    )