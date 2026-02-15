import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_bundle = load_breast_cancer()
X_raw = data_bundle.data
y_target = data_bundle.target

scaler_unit = StandardScaler()
X_scaled = scaler_unit.fit_transform(X_raw)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_target, test_size=0.25, random_state=17
)

clf_lr = LogisticRegression(max_iter=2000)
clf_lr.fit(X_tr, y_tr)

joblib.dump((clf_lr, scaler_unit), "saved_models/logreg.pkl")
