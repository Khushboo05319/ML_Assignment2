import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data_pack = load_breast_cancer()
X_data, y_data = data_pack.data, data_pack.target

X_tr, X_te, y_tr, y_te = train_test_split(
    X_data, y_data, test_size=0.25, random_state=9
)

nb_model = GaussianNB()
nb_model.fit(X_tr, y_tr)
