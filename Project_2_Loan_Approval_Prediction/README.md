
# Loan Approval Prediction Using Machine Learning

## Project Overview

This project develops a supervised machine learning system to predict whether a loan application is likely to be approved based on borrower and loan-related features.

The project focuses on data preprocessing, handling class imbalance, comparing multiple classification models, evaluating model performance, and selecting an appropriate classification threshold for potential deployment.

## Objectives

* Handle missing values
* Encode categorical variables
* Scale numerical features
* Handle class imbalance
* Train and compare multiple machine learning models
* Evaluate models using Precision, Recall, F1 Score and ROC-AUC
* Analyze classification thresholds
* Interpret model results from a business perspective

## Dataset

The dataset contains borrower and loan application information.

Important features include:

* Gender
* Married
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

### Target Variable

`Loan_Status`

* `Y` — Loan Approved
* `N` — Loan Not Approved

## Data Preprocessing

The project uses a preprocessing pipeline containing:

* Median imputation for numerical missing values
* Most-frequent imputation for categorical missing values
* One-hot encoding for categorical variables
* Standard scaling for numerical variables

The `Loan_ID` column is removed because it is an identifier and does not provide useful predictive information.

## Handling Class Imbalance

Class imbalance is handled using balanced class weights.

The following models use:

```python
class_weight="balanced"
```

This gives greater importance to the less frequent class during model training.

## Machine Learning Models

Three classification models are compared:

### 1. Logistic Regression

Used as an interpretable baseline model and for probability-based threshold analysis.

### 2. Decision Tree

Used to capture non-linear relationships between borrower features and loan outcomes.

### 3. Random Forest

An ensemble of decision trees used to capture more complex patterns while reducing the instability of an individual decision tree.

## Evaluation Metrics

The models are evaluated using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

## Threshold Analysis

The default classification threshold is 0.50.

Different thresholds are evaluated to understand the trade-off between precision and recall.

A threshold based on the highest F1 score on the held-out test set is used as an initial operating point.

For real-world deployment, the threshold should be validated using future data, actual business costs, lending policies, calibration analysis, and applicable regulatory and fairness requirements.

## Business Interpretation

The model can be used as a decision-support system for loan applications.

A false positive represents an application predicted as approved when the observed outcome is not approved, while a false negative represents an application predicted as not approved when the observed outcome is approved.

The relative business cost of these errors should be considered when selecting a production threshold.

The model should support, rather than independently replace, organizational lending policies and required review processes.

## Project Structure

```text
Loan_Approval_Prediction/
│
├── Loan_Approval_Prediction.ipynb
├── loan_prediction.csv
└── README.md
```

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Author

Eluri Tejaswini

B.Tech Computer Science and Engineering
