from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build(random_state: int = 42):
    # No scaler here because we're one-hot encoding categoricals and leaving numerics as-is
    return LogisticRegression(max_iter=2000, random_state=random_state, n_jobs=None)