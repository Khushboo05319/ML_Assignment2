import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dataset_obj = load_breast_cancer()
X_vals, y_vals = dataset_obj.data, dataset_obj.target

X_train, X_test, y_train, y_test = train_test_split(
    X_vals, y_vals, test_size=0.25, random_state=23
)

tree_model = DecisionTreeClassifier(
    max_depth=6, criterion="entropy", random_state=23
)
tree_model.fit(X_train, y_train)
