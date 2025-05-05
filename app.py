import streamlit as st
import pickle
import numpy as np

# Load model pipeline
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Kamus untuk input deskriptif → kode numerik
marital_status_map = {
    'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4,
    'Facto Union': 5, 'Legally Separated': 6
}

application_mode_map = {
    '1st phase - general contingent': 1,
    'Ordinance No. 612/93': 2,
    '1st phase - special contingent (Azores Island)': 5,
    'Holders of other higher courses': 7,
    'Ordinance No. 854-B/99': 10,
    'International student (bachelor)': 15,
    '1st phase - special contingent (Madeira Island)': 16,
    '2nd phase - general contingent': 17,
    '3rd phase - general contingent': 18,
    'Over 23 years old': 39,
    'Transfer': 42,
    'Change of course': 43,
    'Change of institution/course': 51,
}

course_map = {
    'Informatics Engineering': 9119,
    'Nursing': 9500,
    'Management': 9147,
    'Tourism': 9254,
    'Communication Design': 9070,
    'Social Service': 9238,
    'Veterinary Nursing': 9085,
    'Biofuel Production Technologies': 33,
    'Oral Hygiene': 9556,
    'Basic Education': 9853,
    'Advertising and Marketing Management': 9670,
    'Animation and Multimedia Design': 171,
}

attendance_map = {'Daytime': 1, 'Evening': 0}
boolean_map = {'Yes': 1, 'No': 0}
gender_map = {'Male': 1, 'Female': 0}

st.title("Student Dropout Prediction")

st.sidebar.header("Input Student Information")

# Sidebar inputs with labels
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
application_mode = st.sidebar.selectbox("Application Mode", list(application_mode_map.keys()))
application_order = st.sidebar.slider("Application Order (0 = 1st choice)", 0, 9, 0)
course = st.sidebar.selectbox("Course", list(course_map.keys()))
attendance = st.sidebar.radio("Class Attendance", list(attendance_map.keys()))
prev_grade = st.sidebar.slider("Previous Qualification Grade", 0, 200, 150)
admission_grade = st.sidebar.slider("Admission Grade", 0, 200, 150)
gender = st.sidebar.radio("Gender", list(gender_map.keys()))
displaced = st.sidebar.radio("Is the student displaced?", list(boolean_map.keys()))
scholarship = st.sidebar.radio("Scholarship Holder?", list(boolean_map.keys()))
special_needs = st.sidebar.radio("Educational Special Needs?", list(boolean_map.keys()))
debtor = st.sidebar.radio("Is the student a debtor?", list(boolean_map.keys()))
fees_up_to_date = st.sidebar.radio("Tuition Fees Up To Date?", list(boolean_map.keys()))
international = st.sidebar.radio("International Student?", list(boolean_map.keys()))
age = st.sidebar.slider("Age at Enrollment", 16, 70, 20)
enrolled_units = st.sidebar.slider("1st Sem: Enrolled Units", 0, 10, 5)
evaluated_units = st.sidebar.slider("1st Sem: Evaluated Units", 0, 10, 5)
approved_units = st.sidebar.slider("1st Sem: Approved Units", 0, 10, 4)
credited_units = st.sidebar.slider("1st Sem: Credited Units", 0, 10, 0)

# Prepare input
input_data = np.array([[
    marital_status_map[marital_status],
    application_mode_map[application_mode],
    application_order,
    course_map[course],
    attendance_map[attendance],
    prev_grade,
    admission_grade,
    gender_map[gender],
    boolean_map[displaced],
    boolean_map[scholarship],
    boolean_map[special_needs],
    boolean_map[debtor],
    boolean_map[fees_up_to_date],
    boolean_map[international],
    age,
    credited_units,
    enrolled_units,
    evaluated_units,
    approved_units
]])

# Mapping hasil prediksi ke label
prediction_map = {
    0: "❌ The student is predicted to **DROP OUT**.",
    1: "📚 The student is predicted to **STAY ENROLLED**.",
    2: "🎓 The student is predicted to **GRADUATE**."
}

if st.button("Predict Dropout Status"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result")
    st.info(prediction_map[int(prediction)])


