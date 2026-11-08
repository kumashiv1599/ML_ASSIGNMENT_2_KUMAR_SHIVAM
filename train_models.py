import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

data=load_breast_cancer(as_frame=True)
X,y=data.data,data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
models={
    "Logistic Regression": Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=2000,random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5,random_state=42),
    "KNN": Pipeline([("scaler",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scaler",StandardScaler()),("model",SVC(kernel="rbf",probability=True,random_state=42))])
}
for name, model in models.items():
    model.fit(X_train,y_train)
    joblib.dump(model,"model/"+name.lower().replace(" ","_")+".joblib")
