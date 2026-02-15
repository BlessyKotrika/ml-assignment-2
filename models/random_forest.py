# models/random_forest.py
from sklearn.ensemble import RandomForestClassifier

def build(n_estimators: int = 150, random_state: int = 42, n_jobs: int = 2, max_depth=None):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,         # was -1; reduce contention in Cloud
        max_depth=max_depth    # keep trees reasonable depth if you want (e.g., 20)
    )