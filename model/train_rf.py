import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

bundle = load_breast_cancer()
X_arr, y_arr = bundle.data, bundle.target

X_tr, X_te, y_tr, y_te = train_test_split(
    X_arr, y_arr, test_size=0.25, random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=42
)
rf_model.fit(X_tr, y_tr)

joblib.dump(rf_model, "saved_models/rf.pkl")
