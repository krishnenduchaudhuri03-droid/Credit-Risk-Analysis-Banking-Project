# Loan_status 1 = Good, 0 = Bad
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("random_forest_credit_model.pkl")
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ["loan_intent","loan_grade","person_home_ownership","cb_person_default_on_file","loan_grade_category"]}

st.title("Credit risk prediction app")
st.write("Enter applicant information to predict that is credit good or bad")

person_age = st.number_input("Person Age", min_value=20, max_value=80, value=25, step=1)
person_income = st.number_input("Person Income", min_value=4000, max_value=6000000, value=50000, step=1000)
person_emp_length = st.number_input("Employment Length (Years)", min_value=0, max_value=60, value=1, step=1)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=0.83, value=0.1, step=0.01)
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=2, max_value=30, value=5, step=1)
loan_amnt = st.number_input("Amount of Loan", min_value=500, max_value=35000, value=5000, step=500)
loan_grade_score = st.number_input("Loan Grade Score", min_value=1, max_value=7, value=1, step=1)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=5.42, max_value=22.22, value=10.0, step=0.01)

loan_intent = st.selectbox("loan_intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT','DEBTCONSOLIDATION'])
loan_grade = st.selectbox("loan_grade",['D', 'B', 'C', 'A', 'E', 'F', 'G'])
cb_person_default_on_file = st.selectbox("cb_person_default_on_file",['Y', 'N'])
person_home_ownership = st.selectbox("person_home_ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_grade_category = st.selectbox("loan_grade_category",['Medium Risk', 'Good Loan', 'Bad Loan'])




loan_grade = loan_grade[0]





input_df = pd.DataFrame({
    "person_age":[person_age],
    "person_income":[person_income],
    "person_emp_length":[person_emp_length],
    "loan_percent_income":[loan_percent_income],
    "cb_person_cred_hist_length":[cb_person_cred_hist_length],
    "loan_amnt":[loan_amnt],
    "loan_grade_score":[loan_grade_score],
    "loan_int_rate":[loan_int_rate],

    "loan_intent": [encoders["loan_intent"].transform([loan_intent])[0]],
    "loan_grade": [encoders["loan_grade"].transform([loan_grade])[0]],
    "person_home_ownership": [encoders["person_home_ownership"].transform([person_home_ownership])[0]],
    "cb_person_default_on_file" : [encoders["cb_person_default_on_file"].transform([cb_person_default_on_file])[0]],
    "loan_grade_category" : [encoders["loan_grade_category"].transform([loan_grade_category])[0]],


})

input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

pred = model.predict(input_df)[0]


if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]


    if pred == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:
        st.error(" The predicted credit risk is: **BAD**")    


