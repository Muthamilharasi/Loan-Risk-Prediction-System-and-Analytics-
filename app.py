# =========================================================
# PROFESSIONAL STREAMLIT UI
# LOAN RISK PREDICTION SYSTEM
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Loan Risk Prediction",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# LOAD MODEL FILES
# =========================================================

model = pickle.load(
    open("loan_risk_model.pkl", "rb")
)

model_columns = pickle.load(
    open("model_columns.pkl", "rb")
)

label_classes = pickle.load(
    open("label_classes.pkl", "rb")
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    background-color: #00ADB5;
    color: white;
    height: 3em;
    border-radius: 10px;
    border: none;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #019CA3;
}

.metric-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("💳 Loan Risk Prediction System")

st.write(
    """
AI-powered financial risk assessment platform
for identifying high-risk loan applicants.
"""
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📊 About Project")

st.sidebar.write(
    """
This system predicts customer loan risk using
Machine Learning and financial analytics.

### Features
- Real-time prediction
- AI-powered analytics
- Risk categorization
- Business insights
"""
)

# =========================================================
# INPUT SECTION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        75,
        30
    )

    job = st.selectbox(
        "Job Level",
        [0, 1, 2, 3],
        help="0 = Unskilled | 3 = Highly Skilled"
    )

    housing = st.selectbox(
        "Housing",
        ["own", "rent", "free"]
    )

    saving_accounts = st.selectbox(
        "Saving Accounts",
        [
            "little",
            "moderate",
            "quite rich",
            "rich"
        ]
    )

with col2:

    checking_account = st.selectbox(
        "Checking Account",
        [
            "little",
            "moderate",
            "rich"
        ]
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=100,
        max_value=50000,
        value=5000
    )

    duration = st.slider(
        "Loan Duration (Months)",
        1,
        72,
        24
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "car",
            "radio/TV",
            "education",
            "business",
            "furniture/equipment",
            "repairs",
            "vacation/others",
            "domestic appliances"
        ]
    )

st.divider()

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔍 Predict Loan Risk"):

    # =====================================================
    # CREATE INPUT DATA
    # =====================================================

    input_data = {

        'age': age,
        'job': job,
        'housing': housing,
        'saving_accounts': saving_accounts,
        'checking_account': checking_account,
        'credit_amount': credit_amount,
        'duration': duration,
        'purpose': purpose

    }

    input_df = pd.DataFrame([input_data])

    # =====================================================
    # ONE HOT ENCODING
    # =====================================================

    input_encoded = pd.get_dummies(
        input_df
    )

    # =====================================================
    # MATCH TRAINING COLUMNS
    # =====================================================

    final_df = pd.DataFrame(
        columns=model_columns
    )

    for col in final_df.columns:

        if col in input_encoded.columns:
            final_df[col] = input_encoded[col]

    final_df = final_df.fillna(0)

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(
        final_df
    )

    predicted_label = label_classes[
        prediction[0]
    ]

    probabilities = model.predict_proba(
        final_df
    )

    confidence = (
        np.max(probabilities) * 100
    )

    # =====================================================
    # BUSINESS RULES
    # =====================================================

    risk_score = 0

    if credit_amount > 15000:
        risk_score += 2

    if duration > 48:
        risk_score += 2

    if saving_accounts == 'little':
        risk_score += 2

    if checking_account == 'little':
        risk_score += 2

    if age < 25:
        risk_score += 1

    # =====================================================
    # FINAL RESULT
    # =====================================================

    if risk_score >= 5:
        predicted_label = "High Risk"

    elif risk_score >= 3:
        predicted_label = "Medium Risk"

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    st.subheader("📈 Prediction Result")

    if predicted_label == "High Risk":

        st.error(
            f"⚠️ High Risk Customer ({confidence:.2f}% confidence)"
        )

    elif predicted_label == "Medium Risk":

        st.warning(
            f"🟠 Medium Risk Customer ({confidence:.2f}% confidence)"
        )

    else:

        st.success(
            f"✅ Low Risk Customer ({confidence:.2f}% confidence)"
        )

    # =====================================================
    # METRICS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Credit Amount",
            f"${credit_amount}"
        )

    with c2:
        st.metric(
            "Loan Duration",
            f"{duration} Months"
        )

    with c3:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    # =====================================================
    # PROBABILITY BREAKDOWN
    # =====================================================

    st.subheader("📊 Prediction Probabilities")

    for i, label in enumerate(label_classes):

        st.progress(
            float(probabilities[0][i])
        )

        st.write(
            f"{label}: {probabilities[0][i] * 100:.2f}%"
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Developed using Python, Machine Learning, and Streamlit"
)