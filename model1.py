import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, recall_score, f1_score, make_scorer


def load_credit_card_fraud_model():
    data = pd.read_csv('/Users/chiragn/Documents/Hackathon/credit_card_fraud.csv')
    data = data.drop(['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], axis=1)
    for column in ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']:
        data[column].replace(0, data[column].mean(), inplace=True)

    X = data.drop('default payment next month', axis=1)
    y = data['default payment next month']

    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sm)

    xgb_model = XGBClassifier()
    xgb_model.fit(X_scaled, y_sm)
    
    return xgb_model, scaler, X.columns, data, X_scaled, y_sm


st.title('Credit Card Fraud Detection')


st.write("""
### Introduction
Credit card fraud is a significant concern for both cardholders and financial institutions. Detecting fraudulent transactions quickly is crucial to prevent financial loss and protect customers. 

This application uses a machine learning model to predict whether a credit card transaction is fraudulent or not. By analyzing various transaction features, the model provides a prediction that helps in identifying potentially fraudulent activities.

### About This Model
The model used in this application is a supervised machine learning model based on the XGBoost classifier. It is trained on a dataset of credit card transactions where each transaction is labeled as either fraudulent or non-fraudulent.

### Feature Descriptions
- **PAY_0**: Repayment status in September.
- **PAY_2**: Repayment status in August.
- **PAY_3**: Repayment status in July.
- **PAY_4**: Repayment status in June.
- **PAY_5**: Repayment status in May.
- **PAY_6**: Repayment status in April.

- **PAY_AMT1**: Previous payment amount in September.
- **PAY_AMT2**: Previous payment amount in August.
- **PAY_AMT3**: Previous payment amount in July.
- **PAY_AMT4**: Previous payment amount in June.
- **PAY_AMT5**: Previous payment amount in May.
- **PAY_AMT6**: Previous payment amount in April.

The model employs several techniques to improve its accuracy:
- **Data Preprocessing**: Handles missing values and scales features for better performance.
- **SMOTE**: A technique to handle class imbalance by generating synthetic samples for the minority class.
- **Feature Scaling**: Standardizes features to ensure that the model performs optimally.

The model is evaluated using cross-validation to ensure its robustness and reliability. Performance metrics such as accuracy, recall, and F1 score are used to gauge its effectiveness.

### How to Use This Application
To use this application, enter the transaction details in the fields below, and the model will predict whether the transaction is likely to be fraudulent.

""")


xgb_model, scaler, feature_names, data, X_scaled, y_sm = load_credit_card_fraud_model()

st.subheader('Predict if you will default on payment next month')

user_input = {}
for column in feature_names:
    min_value = float(data[column].min())
    max_value = float(data[column].max())
    step = (max_value - min_value) / 100
    
    description = {
        'PAY_0': 'Repayment status in Sep',
        'PAY_2': 'Repayment status in Aug',
        'PAY_3': 'Repayment status in Jul',
        'PAY_4': 'Repayment status in Jun',
        'PAY_5': 'Repayment status in May',
        'PAY_6': 'Repayment status in Apr',
        'PAY_AMT1': 'Payment amount in Sep',
        'PAY_AMT2': 'Payment amount in Aug',
        'PAY_AMT3': 'Payment amount in Jul',
        'PAY_AMT4': 'Payment amount in Jun',
        'PAY_AMT5': 'Payment amount in May',
        'PAY_AMT6': 'Payment amount in Apr'
    }
    
    slider_label = f"{column}: {description.get(column, column)}"
    
    user_input[column] = st.slider(
        slider_label,
        min_value=min_value,
        max_value=max_value,
        value=float(data[column].mean()),
        step=step
    )

user_df = pd.DataFrame([user_input])
user_df = user_df.astype(float)
user_scaled = scaler.transform(user_df)

prediction = xgb_model.predict(user_scaled)


if prediction[0] == 0:
    st.markdown("### **Prediction: Not likely to default on payment next month.**")
    st.markdown("""
    **Possible reasons:**
    - Good credit history
    - Timely payments
    - Low debt
    """)
else:
    st.markdown("### **Prediction: Likely to default on payment next month.**")
    st.markdown("""
    **Possible reasons:**
    - Late payments
    - High outstanding balance
    - Multiple credit inquiries
    """)


stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'accuracy': make_scorer(accuracy_score), 'recall': make_scorer(recall_score), 'f1_score': make_scorer(f1_score)}
cv_results = cross_validate(xgb_model, X_scaled, y_sm, cv=stratified_kfold, scoring=scoring)

st.write("## Model Performance")
st.write(f"**Cross-Validation Accuracy Scores:** {cv_results['test_accuracy']}")
st.write(f"**Mean Cross-Validation Accuracy:** {cv_results['test_accuracy'].mean():.4f}")
st.write(f"**Recall Scores:** {cv_results['test_recall']}")
st.write(f"**Mean Recall Score:** {cv_results['test_recall'].mean():.4f}")
st.write(f"**F1 Scores:** {cv_results['test_f1_score']}")
st.write(f"**Mean F1 Score:** {cv_results['test_f1_score'].mean():.4f}")
