import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="ML Classifier Lab", layout="wide")

st.title("🧠 Breast Cancer Classification Lab")

uploaded_csv = st.file_uploader("Upload CSV (Test Data Only)", type=["csv"])

model_map = {
    "Logistic Regression": "saved_models/logreg.pkl",
    "Decision Tree": "saved_models/tree.pkl",
    "k-NN": "saved_models/knn.pkl",
    "Naive Bayes": "saved_models/nb.pkl",
    "Random Forest": "saved_models/rf.pkl",
    "XGBoost": "saved_models/xgb.pkl"
}

model_choice = st.selectbox("Choose Model", list(model_map.keys()))

if uploaded_csv is not None:
    df_test = pd.read_csv(uploaded_csv)
    X_input = df_test.iloc[:, :-1]
    y_true = df_test.iloc[:, -1]

    model_obj = joblib.load(model_map[model_choice])

    if isinstance(model_obj, tuple):
        model, scaler = model_obj
        X_input = scaler.transform(X_input)
    else:
        model = model_obj

    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 3))
    col1.metric("AUC", round(roc_auc_score(y_true, y_prob), 3))

    col2.metric("Precision", round(precision_score(y_true, y_pred), 3))
    col2.metric("Recall", round(recall_score(y_true, y_pred), 3))

    col3.metric("F1 Score", round(f1_score(y_true, y_pred), 3))
    col3.metric("MCC", round(matthews_corrcoef(y_true, y_pred), 3))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_true, y_pred))

