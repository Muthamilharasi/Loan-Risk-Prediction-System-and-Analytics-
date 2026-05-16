import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================
# LOAD MODEL & COLUMNS
# =====================================

model = pickle.load(
    open("risk_model.pkl", "rb")
)

model_columns = pickle.load(
    open("model_columns.pkl", "rb")
)

# =====================================
# TITLE
# =====================================

st.title("Loan Risk Prediction")

# =====================================
# USER INPUTS
# =====================================

credit_score = st.number_input(
    "Credit Score",
    value=700.0
)

annual_income = st.number_input(
    "Annual Income",
    value=50000.0
)

monthly_debt = st.number_input(
    "Monthly Debt",
    value=1000.0
)

years_of_credit_history = st.number_input(
    "Years of Credit History",
    value=10.0
)

dti_ratio = st.number_input(
    "Debt-To-Income Ratio",
    value=0.3
)

# =====================================
# PREDICTION
# =====================================

if st.button("Predict Risk"):

    # Create dataframe with ALL columns
    input_data = pd.DataFrame(
        np.zeros((1, len(model_columns))),
        columns=model_columns
    )

    # Fill important features
    if 'credit_score' in input_data.columns:
        input_data['credit_score'] = credit_score

    if 'annual_income' in input_data.columns:
        input_data['annual_income'] = annual_income

    if 'monthly_debt' in input_data.columns:
        input_data['monthly_debt'] = monthly_debt

    if 'years_of_credit_history' in input_data.columns:
        input_data['years_of_credit_history'] = years_of_credit_history

    if 'dti_ratio' in input_data.columns:
        input_data['dti_ratio'] = dti_ratio

    # Prediction
    prediction = model.predict(input_data)

    # Output
    if prediction[0] == 0:
        st.success("Low Risk Customer")

    elif prediction[0] == 1:
        st.warning("Medium Risk Customer")

    else:
        st.error("High Risk Customer")