# models/random_forest.py
from sklearn.ensemble import RandomForestClassifier

def build(n_estimators: int = 300, random_state: int = 42, n_jobs: int = -1):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )