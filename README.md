# ML_Assignment2
Breast cancer detection model - classification model
**Problem Statement**

Comparing classical and ensemble machine learning classifiers for breast cancer diagnosis using standard evaluation metrics.

**Dataset Description**

The Breast Cancer Wisconsin Diagnostic dataset contains 569 patient records with 30 numerical features computed from digitized FNA images.

**Model Comparison Table**
Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.98	0.99	0.98	0.99	0.99	0.96
Decision Tree	0.92	0.91	0.93	0.91	0.92	0.84
kNN	0.95	0.97	0.96	0.95	0.95	0.90
Naive Bayes	0.94	0.96	0.94	0.95	0.94	0.88
Random Forest	0.97	0.99	0.97	0.98	0.98	0.94
XGBoost	0.98	0.99	0.98	0.99	0.99	0.96

**Observations**
|Model|	Observation|
|Logistic Regression|	Excellent baseline with strong generalization|
|Decision Tree|	Easy to interpret but prone to overfitting|
|kNN|	Performs well but sensitive to scaling|
|Naive Bayes	|Fast and robust with independence assumption|
|Random Forest|	Handles feature interactions effectively|
|XGBoost|	Best overall performance with minimal bias|
