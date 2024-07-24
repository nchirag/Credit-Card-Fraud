# Credit-Card-Fraud
Introduction
Credit card fraud is a significant concern for both cardholders and financial institutions. Detecting fraudulent transactions quickly is crucial to prevent financial loss and protect customers. This application uses a machine learning model to predict whether a credit card transaction is fraudulent or not. By analyzing various transaction features, the model provides a prediction that helps in identifying potentially fraudulent activities.

About This Model
The model used in this application is a supervised machine learning model based on the XGBoost classifier. It is trained on a dataset of credit card transactions where each transaction is labeled as either fraudulent or non-fraudulent.

Features Used in the Model
The dataset contains various features related to credit card transactions, such as:

BILL_AMT1-6: Amount of bill statement in various months.
PAY_0-6: History of past payments.
AGE: Age of the cardholder.
default payment next month: Whether the cardholder defaulted on payment next month.
Data Preprocessing
Dropping Unnecessary Columns: Columns like LIMIT_BAL, SEX, EDUCATION, MARRIAGE, and PAY_AMT1-6 were dropped.
Handling Missing Values: Missing values in the BILL_AMT columns were replaced with the mean of the respective columns.
Feature Scaling: Features were scaled using StandardScaler.
Handling Imbalanced Data: The SMOTE technique was used to handle class imbalance in the target variable.
Model Training
The model is trained using the XGBClassifier from the XGBoost library. The training involves:

Splitting the data using StratifiedKFold for cross-validation.
Evaluating model performance using accuracy, recall, and F1 score.
