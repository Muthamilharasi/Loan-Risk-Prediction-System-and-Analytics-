# =========================================================
# TRAIN MODEL FOR LOAN RISK PREDICTION
# =========================================================

import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# =========================================================
# LOAD DATASET
# =========================================================

loan_data = pd.read_csv(
    "german_credit_data.csv"
)

print("Dataset Loaded Successfully")

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

loan_data.columns = (
    loan_data.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)

# =========================================================
# REMOVE EXTRA COLUMN
# =========================================================

if 'unnamed:_0' in loan_data.columns:

    loan_data.drop(
        'unnamed:_0',
        axis=1,
        inplace=True
    )

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

loan_data['saving_accounts'].fillna(
    'little',
    inplace=True
)

loan_data['checking_account'].fillna(
    'little',
    inplace=True
)

# =========================================================
# CREATE TARGET VARIABLE
# =========================================================

def classify_risk(row):

    risk_score = 0

    # Credit Amount Risk
    if row['credit_amount'] > 15000:
        risk_score += 3

    elif row['credit_amount'] > 10000:
        risk_score += 2

    elif row['credit_amount'] > 7000:
        risk_score += 1

    # Duration Risk
    if row['duration'] > 48:
        risk_score += 3

    elif row['duration'] > 36:
        risk_score += 2

    elif row['duration'] > 24:
        risk_score += 1

    # Savings Risk
    if row['saving_accounts'] == 'little':
        risk_score += 2

    # Checking Account Risk
    if row['checking_account'] == 'little':
        risk_score += 2

    # Young Age Risk
    if row['age'] < 25:
        risk_score += 1

    # Housing Risk
    if row['housing'] == 'rent':
        risk_score += 1

    # Job Risk
    if row['job'] == 0:
        risk_score += 1

    # FINAL LABEL
    if risk_score >= 7:
        return 'High Risk'

    elif risk_score >= 4:
        return 'Medium Risk'

    else:
        return 'Low Risk'

loan_data['risk_category'] = loan_data.apply(
    classify_risk,
    axis=1
)

print("\nRisk Distribution:")
print(
    loan_data['risk_category']
    .value_counts()
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

loan_data['monthly_payment'] = (
    loan_data['credit_amount']
    /
    loan_data['duration']
)

loan_data['credit_age_ratio'] = (
    loan_data['credit_amount']
    /
    loan_data['age']
)

loan_data['long_term_loan'] = np.where(
    loan_data['duration'] > 36,
    1,
    0
)

loan_data['high_credit'] = np.where(
    loan_data['credit_amount'] > 10000,
    1,
    0
)

# =========================================================
# LABEL ENCODING TARGET
# =========================================================

label_encoder = LabelEncoder()

loan_data['risk_encoded'] = (
    label_encoder.fit_transform(
        loan_data['risk_category']
    )
)

# Save Label Classes
pickle.dump(
    label_encoder.classes_,
    open('label_classes.pkl', 'wb')
)

# =========================================================
# FEATURES & TARGET
# =========================================================

X = loan_data.drop(
    ['risk_category', 'risk_encoded'],
    axis=1
)

y = loan_data['risk_encoded']

# =========================================================
# ONE HOT ENCODING
# =========================================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# =========================================================
# SAVE MODEL COLUMNS
# =========================================================

pickle.dump(
    X.columns,
    open('model_columns.pkl', 'wb')
)

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save Scaler
pickle.dump(
    scaler,
    open('scaler.pkl', 'wb')
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# SMOTE BALANCING
# =========================================================

smote = SMOTE(
    random_state=42
)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    max_depth=10,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight='balanced'
)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(
    X_train,
    y_train
)

# =========================================================
# EVALUATE MODEL
# =========================================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nModel Accuracy: {accuracy:.4f}")

# =========================================================
# SAVE MODEL
# =========================================================

pickle.dump(
    model,
    open('loan_risk_model.pkl', 'wb')
)

print("\nModel Saved Successfully!")

print("\nCreated Files:")
print("- loan_risk_model.pkl")
print("- scaler.pkl")
print("- model_columns.pkl")
print("- label_classes.pkl")