import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="WDBC Classifier Lab", page_icon="🔬", layout="wide")
st.title("🔬 WDBC Classification Model Explorer")
st.caption("Upload held-out test data, choose a saved classifier, and evaluate it without retraining.")
MODELS={"Logistic Regression": "logistic_regression.joblib", "Decision Tree": "decision_tree.joblib", "KNN": "knn.joblib", "Naive Bayes": "naive_bayes.joblib", "Support Vector Machine": "support_vector_machine.joblib"}
TARGET="diagnosis"
LABEL_MAP={"malignant":0,"benign":1,"M":0,"B":1,0:0,1:1}
file=st.file_uploader("Upload test CSV", type=["csv"])
model_name=st.selectbox("Select model", list(MODELS))
if file is not None:
    df=pd.read_csv(file)
    if TARGET not in df.columns:
        st.error("CSV must contain the target column: diagnosis")
        st.stop()
    y=df[TARGET].map(LABEL_MAP)
    X=df.drop(columns=[TARGET])
    if y.isna().any():
        st.error("Target values must be malignant/benign, M/B, or 0/1.")
        st.stop()
    model=joblib.load("model/"+MODELS[model_name])
    expected=list(model.feature_names_in_) if hasattr(model,"feature_names_in_") else list(X.columns)
    missing=[c for c in expected if c not in X.columns]
    if missing:
        st.error("Missing required feature columns: "+", ".join(missing))
        st.stop()
    X=X[expected]
    pred=model.predict(X)
    prob=model.predict_proba(X)[:,1]
    vals={"Accuracy":accuracy_score(y,pred),"AUC":roc_auc_score(y,prob),"Precision":precision_score(y,pred),"Recall":recall_score(y,pred),"F1":f1_score(y,pred),"MCC":matthews_corrcoef(y,pred)}
    cols=st.columns(6)
    for c,(k,v) in zip(cols,vals.items()): c.metric(k,f"{v:.4f}")
    st.subheader("Confusion Matrix")
    cm=confusion_matrix(y,pred)
    cm_df=pd.DataFrame(cm,index=["Actual malignant","Actual benign"],columns=["Pred malignant","Pred benign"])
    st.markdown(cm_df.to_html(), unsafe_allow_html=True)
    st.subheader("Classification Report")
    report_df=pd.DataFrame(classification_report(y,pred,target_names=["malignant","benign"],output_dict=True)).T.round(4)
    st.markdown(report_df.to_html(), unsafe_allow_html=True)
    st.info("The uploaded test data is used only for evaluation. The selected model is loaded from disk and is not retrained.")
else:
    st.info("Upload the provided test_data.csv to begin.")
