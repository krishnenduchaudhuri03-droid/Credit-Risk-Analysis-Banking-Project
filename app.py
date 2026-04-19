import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Decision Engine",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

model = load_model()

# -------------------------------
# ENCODING MAPS
# -------------------------------
home_map    = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}
default_map = {"No": 0, "Yes": 1}
grade_map   = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
purpose_map = {
    "Personal": 0, "Education": 1, "Medical": 2,
    "Business": 3, "Home": 4, "Auto": 5
}

# -------------------------------
# HEADER
# -------------------------------
st.title("🏦 Credit Risk Decision Engine")
st.markdown("### AI-powered Loan Default Prediction System")
st.markdown("---")

# -------------------------------
# FORM
# -------------------------------
with st.form("credit_risk_form"):

    st.subheader("👤 Applicant Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age              = st.number_input("Age", 18, 75, 30)
        income           = st.number_input("Annual Income (₹)", 10_000, 10_000_000, 500_000)

    with col2:
        employment_years = st.number_input("Employment Length (Years)", 0, 40, 5)
        home_ownership   = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

    with col3:
        prior_default    = st.selectbox(
            "Prior Default History",
            ["No", "Yes"],
            help="Has the applicant defaulted on a loan before?"
        )

    st.markdown("---")
    st.subheader("💰 Loan Details")
    col4, col5, col6 = st.columns(3)

    with col4:
        loan_amount      = st.number_input("Loan Amount (₹)", 1_000, 5_000_000, 200_000)
        interest_rate    = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.0)

    with col5:
        loan_grade       = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_purpose     = st.selectbox(
            "Loan Purpose",
            ["Personal", "Education", "Medical", "Business", "Home", "Auto"]
        )

    with col6:
        credit_history_length = st.number_input("Credit History Length (Years)", 0, 30, 5)

    submit = st.form_submit_button("🚀 Predict Credit Risk")

# -------------------------------
# PREDICTION (only on submit)
# -------------------------------
if submit:

    # --- Build raw input ---
    loan_percent_income = loan_amount / income

    raw = {
        "person_age":                  age,
        "person_income":               income,
        "person_home_ownership":       home_map[home_ownership],
        "person_emp_length":           employment_years,
        "loan_intent":                 purpose_map[loan_purpose],
        "loan_grade":                  grade_map[loan_grade],
        "loan_amnt":                   loan_amount,
        "loan_int_rate":               interest_rate,
        "loan_percent_income":         loan_percent_income,
        "cb_person_cred_hist_length":  credit_history_length,
        "prior_default":               default_map[prior_default],
    }

    input_df = pd.DataFrame([raw])

    # --- Feature Engineering ---
    input_df["person_emp_length_missing"] = input_df["person_emp_length"].isnull().astype(int)
    input_df["loan_int_rate_missing"]     = input_df["loan_int_rate"].isnull().astype(int)
    input_df["log_income"]                = np.log1p(input_df["person_income"])
    input_df["dti_ratio"]                 = input_df["loan_amnt"] / input_df["person_income"]
    input_df["age_emp_ratio"]             = input_df["person_age"] / (input_df["person_emp_length"] + 1)
    input_df["loan_per_hist_year"]        = input_df["loan_amnt"] / (input_df["cb_person_cred_hist_length"] + 1)
    input_df["grade_score"]               = input_df["loan_grade"]
    input_df["loan_risk_tier"]            = pd.cut(
        input_df["loan_percent_income"],
        bins=[0, 0.1, 0.2, 0.3, 1],
        labels=[1, 2, 3, 4]
    ).astype(int)

    # --- Predict ---
    try:
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # --- Output ---
        st.markdown("---")
        st.subheader("📊 Risk Decision Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.error("❌ High Risk — Reject Loan")
            else:
                st.success("✅ Low Risk — Approve Loan")

        with col2:
            st.metric("Default Probability", f"{probability:.2%}")

        with col3:
            if probability > 0.7:
                st.error("Very High Risk")
            elif probability > 0.4:
                st.warning("Moderate Risk")
            else:
                st.success("Low Risk")

        st.markdown("### 🏦 Final Decision")
        if probability > 0.6:
            st.error("🚫 Loan Rejected")
        else:
            st.success("✅ Loan Approved")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.info("Check that `credit_risk_model.pkl` expects these features and is compatible with this input.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built for Banking-Grade Credit Risk Assessment | Streamlit App")
