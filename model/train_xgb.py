import joblib
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
X_feat, y_lbl = dataset.data, dataset.target

X_tr, X_te, y_tr, y_te = train_test_split(
    X_feat, y_lbl, test_size=0.25, random_state=99
)

xgb_model = xgb.XGBClassifier(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    eval_metric="logloss"
)

xgb_model.fit(X_tr, y_tr)

joblib.dump(xgb_model, "saved_models/xgb.pkl")
