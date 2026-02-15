import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

import xgboost as xgb

st.set_page_config(page_title="ML Classification Lab", layout="wide")
st.title("🧠 Breast Cancer Classification – BITS ML Assignment")

# -------------------------------
# Dataset loading (cached)
# -------------------------------
@st.cache_resource
def load_and_prepare_data():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names

    scaler_obj = StandardScaler()
    X_scaled = scaler_obj.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.25, random_state=21
    )

    return X_tr, X_te, y_tr, y_te, scaler_obj, feature_names

X_train, X_test, y_train, y_test, scaler, feature_labels = load_and_prepare_data()

# -------------------------------
# Model factory (cached)
# -------------------------------
@st.cache_resource
def build_model(model_key):
    if model_key == "Logistic Regression":
        model = LogisticRegression(max_iter=1500)

    elif model_key == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=6, criterion="entropy", random_state=21
        )

    elif model_key == "k-NN":
        model = KNeighborsClassifier(n_neighbors=7)

    elif model_key == "Naive Bayes":
        model = GaussianNB()

    elif model_key == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=120, max_depth=8, random_state=21
        )

    elif model_key == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            eval_metric="logloss"
        )

    model.fit(X_train, y_train)
    return model

# -------------------------------
# UI Controls
# -------------------------------
model_name = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "k-NN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.file_uploader(
    "Upload Test CSV (Last column = target)",
    type=["csv"]
)

# -------------------------------
# Evaluation
# -------------------------------
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    X_user = test_df.iloc[:, :-1]
    y_actual = test_df.iloc[:, -1]

    X_user_scaled = scaler.transform(X_user)

    model_instance = build_model(model_name)

    y_predicted = model_instance.predict(X_user_scaled)
    y_prob = model_instance.predict_proba(X_user_scaled)[:, 1]

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y_actual, y_predicted), 3))
    col1.metric("AUC", round(roc_auc_score(y_actual, y_prob), 3))

    col2.metric("Precision", round(precision_score(y_actual, y_predicted), 3))
    col2.metric("Recall", round(recall_score(y_actual, y_predicted), 3))

    col3.metric("F1 Score", round(f1_score(y_actual, y_predicted), 3))
    col3.metric("MCC", round(matthews_corrcoef(y_actual, y_predicted), 3))

    st.subheader("Confusion Matrix")
    st.dataframe(confusion_matrix(y_actual, y_predicted))
