# WDBC Classification Model Explorer

## a. Problem statement
Build and compare five machine-learning classifiers that predict whether a breast mass is malignant or benign from numeric features extracted from digitized fine-needle-aspirate images. The same held-out test set is used for all five models.

## b. Dataset description
Dataset: Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository. It contains 569 instances and 30 real-valued input features, with no missing values in the UCI listing. The target has two classes: malignant and benign. This satisfies the assignment minimum of 500 instances and 12 features.

Dataset source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Split used: stratified 80% training / 20% testing, `random_state=42`. Scaling is kept inside pipelines for Logistic Regression, KNN and SVM, preventing test-set fitting.

## c. GitHub Repository Link
**Repository status:** A complete local Git repository with commit history is included in the ZIP package. A public GitHub URL requires publishing through the student's authenticated GitHub account.

## Live Streamlit App
**Application status:** The Streamlit app has been run and verified locally. A public Streamlit URL requires deployment through the student's authenticated Streamlit Community Cloud account.

## d. Models used and comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline after standardization; high discrimination and balanced classification performance. |
| Decision Tree | Interpretable nonlinear model, but a single tree is more sensitive to sampling variation and can overfit compared with regularized or margin-based models. |
| KNN | Performs strongly after scaling because distance calculations are otherwise dominated by large-magnitude measurements such as area and perimeter. |
| Naive Bayes | Fast probabilistic baseline. Its conditional-independence assumption is restrictive because several tumor measurements are correlated. |
| Support Vector Machine | RBF SVM benefits from standardized continuous features and can represent nonlinear decision boundaries; it is a strong fit for this moderate-sized numeric dataset. |
| **Overall winner for this experiment** | **Logistic Regression**, selected primarily by F1, with AUC and MCC used as supporting metrics. |

## Streamlit functionality
- Upload a CSV containing all 30 feature columns and the `diagnosis` target.
- Select any of the five saved models.
- Display Accuracy, AUC, Precision, Recall, F1 and MCC.
- Display confusion matrix and classification report.
- Models are loaded from `.joblib` files and are **not retrained** on uploaded test data.

## Repository files
```text
.
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Run locally / on BITS Virtual Lab
```bash
pip install -r requirements.txt
streamlit run app.py
```
Take one screenshot clearly showing execution on BITS Virtual Lab for the final PDF.

## Deployment
Push these files to GitHub, then create a Streamlit Community Cloud app using the repository, `main` branch and `app.py`. Test the live URL with `test_data.csv` before submission.
