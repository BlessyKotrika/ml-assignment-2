from sklearn.tree import DecisionTreeClassifier

def build(random_state: int = 42):
    return DecisionTreeClassifier(random_state=random_state)