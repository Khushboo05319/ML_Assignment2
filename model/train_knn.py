import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_obj = load_breast_cancer()
X, y = data_obj.data, data_obj.target

scaler_knn = StandardScaler()
X_scaled = scaler_knn.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.25, random_state=11)

knn_clf = KNeighborsClassifier(n_neighbors=7, metric="minkowski")
knn_clf.fit(X_tr, y_tr)
