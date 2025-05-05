import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Dictionaries for categorical mapping
marital_status_map = {'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4, 'Facto Union': 5, 'Legally Separated': 6}
application_mode_map = {
    '1st phase - general contingent': 1, 'Ordinance No. 612/93': 2,
    '1st phase - special contingent (Azores Island)': 5, 'Holders of other higher courses': 7,
    'Ordinance No. 854-B/99': 10, 'International student (bachelor)': 15,
    '1st phase - special contingent (Madeira Island)': 16, '2nd phase - general contingent': 17,
    '3rd phase - general contingent': 18, 'Over 23 years old': 39,
    'Transfer': 42, 'Change of course': 43, 'Change of institution/course': 51,
}
course_map = {
    'Informatics Engineering': 9119, 'Nursing': 9500, 'Management': 9147, 'Tourism': 9254,
    'Communication Design': 9070, 'Social Service': 9238, 'Veterinary Nursing': 9085,
    'Biofuel Production Technologies': 33, 'Oral Hygiene': 9556, 'Basic Education': 9853,
    'Advertising and Marketing Management': 9670, 'Animation and Multimedia Design': 171,
}
attendance_map = {'Daytime': 1, 'Evening': 0}
boolean_map = {'Yes': 1, 'No': 0}
gender_map = {'Male': 1, 'Female': 0}

st.title("🎓 Student Dropout Prediction")
st.sidebar.header("Input Student Information")

# Input fields (ordered as required)
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
application_mode = st.sidebar.selectbox("Application Mode", list(application_mode_map.keys()))
application_order = st.sidebar.slider("Application Order (0 = 1st choice)", 0, 9, 0)
course = st.sidebar.selectbox("Course", list(course_map.keys()))
attendance = st.sidebar.radio("Class Attendance", list(attendance_map.keys()))
prev_qualification = st.sidebar.slider("Previous Qualification (Code)", 0, 100, 1)
prev_grade = st.sidebar.slider("Previous Qualification Grade", 0, 200, 150)
nacionality = st.sidebar.slider("Nationality (Code)", 1, 250, 1)
mother_edu = st.sidebar.slider("Mother's Qualification (Code)", 0, 100, 1)
father_edu = st.sidebar.slider("Father's Qualification (Code)", 0, 100, 1)
mother_job = st.sidebar.slider("Mother's Occupation (Code)", 0, 100, 1)
father_job = st.sidebar.slider("Father's Occupation (Code)", 0, 100, 1)
admission_grade = st.sidebar.slider("Admission Grade", 0, 200, 150)
displaced = st.sidebar.radio("Is the student displaced?", list(boolean_map.keys()))
special_needs = st.sidebar.radio("Educational Special Needs?", list(boolean_map.keys()))
debtor = st.sidebar.radio("Is the student a debtor?", list(boolean_map.keys()))
fees_up_to_date = st.sidebar.radio("Tuition Fees Up To Date?", list(boolean_map.keys()))
gender = st.sidebar.radio("Gender", list(gender_map.keys()))
scholarship = st.sidebar.radio("Scholarship Holder?", list(boolean_map.keys()))
age = st.sidebar.slider("Age at Enrollment", 16, 70, 20)
international = st.sidebar.radio("International Student?", list(boolean_map.keys()))

# 1st Sem
cred_1st = st.sidebar.slider("1st Sem: Credited Units", 0, 10, 0)
enr_1st = st.sidebar.slider("1st Sem: Enrolled Units", 0, 10, 5)
eval_1st = st.sidebar.slider("1st Sem: Evaluated Units", 0, 10, 5)
appr_1st = st.sidebar.slider("1st Sem: Approved Units", 0, 10, 4)
grade_1st = st.sidebar.slider("1st Sem: Average Grade", 0, 20, 14)
no_eval_1st = st.sidebar.slider("1st Sem: Units Without Evaluation", 0, 10, 0)

# 2nd Sem
cred_2nd = st.sidebar.slider("2nd Sem: Credited Units", 0, 10, 0)
enr_2nd = st.sidebar.slider("2nd Sem: Enrolled Units", 0, 10, 5)
eval_2nd = st.sidebar.slider("2nd Sem: Evaluated Units", 0, 10, 5)
appr_2nd = st.sidebar.slider("2nd Sem: Approved Units", 0, 10, 4)
grade_2nd = st.sidebar.slider("2nd Sem: Average Grade", 0, 20, 14)
no_eval_2nd = st.sidebar.slider("2nd Sem: Units Without Evaluation", 0, 10, 0)

# Economic context
unemployment_rate = st.sidebar.slider("Unemployment Rate (%)", 0.0, 20.0, 7.0)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 2.0)
gdp = st.sidebar.slider("GDP Growth Rate (%)", -10.0, 10.0, 1.5)

# Arrange inputs in the correct order
input_data = np.array([[
    marital_status_map[marital_status],
    application_mode_map[application_mode],
    application_order,
    course_map[course],
    attendance_map[attendance],
    prev_qualification,
    prev_grade,
    nacionality,
    mother_edu,
    father_edu,
    mother_job,
    father_job,
    admission_grade,
    boolean_map[displaced],
    boolean_map[special_needs],
    boolean_map[debtor],
    boolean_map[fees_up_to_date],
    gender_map[gender],
    boolean_map[scholarship],
    age,
    boolean_map[international],
    cred_1st,
    enr_1st,
    eval_1st,
    appr_1st,
    grade_1st,
    no_eval_1st,
    cred_2nd,
    enr_2nd,
    eval_2nd,
    appr_2nd,
    grade_2nd,
    no_eval_2nd,
    unemployment_rate,
    inflation_rate,
    gdp
]])

# Mapping prediction output to label
prediction_map = {
    0: "❌ The student is predicted to **DROP OUT**.",
    1: "📚 The student is predicted to **STAY ENROLLED**.",
    2: "🎓 The student is predicted to **GRADUATE**."
}

if st.button("Predict Dropout Status"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result")
    st.info(prediction_map[int(prediction)])
