from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def build(n_neighbors: int = 5):
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors))
    ])