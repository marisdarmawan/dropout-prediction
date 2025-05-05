import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Label mapping
label_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

# Dictionary untuk pilihan input kategori
choices = {
    "Marital status": {
        1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto union", 6: "Legally separated"
    },
    "Application mode": {
        1: "1st phase - general contingent", 2: "Ordinance No. 612/93", 5: "1st phase - special (Azores)", 
        7: "Other higher courses", 10: "Ordinance No. 854-B/99", 15: "International student", 
        16: "1st phase - Madeira", 17: "2nd phase", 18: "3rd phase", 26: "Different Plan", 
        27: "Other Institution", 39: "Over 23 years old", 42: "Transfer", 43: "Change of course", 
        44: "Tech diploma", 51: "Change institution/course", 53: "Short cycle diploma", 
        57: "Change institution/course (Int’l)"
    },
    "Course": {
        33: "Biofuel", 171: "Animation", 8014: "Social Service (evening)", 9003: "Agronomy",
        9070: "Comm Design", 9085: "Vet Nursing", 9119: "Informatics Eng.", 9130: "Equinculture", 
        9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 
        9556: "Oral Hygiene", 9670: "Marketing", 9773: "Journalism", 9853: "Basic Ed", 
        9991: "Management (evening)"
    },
    "Daytime/evening attendance": {0: "Evening", 1: "Daytime"},
    "Previous qualification": {
        1: "Secondary", 2: "Bachelor", 3: "Degree", 4: "Master", 5: "Doctorate", 
        6: "Attending Higher Ed", 9: "12th not completed", 10: "11th not completed",
        12: "Other 11th", 14: "10th", 15: "10th not completed", 19: "Basic 3rd cycle",
        38: "Basic 2nd cycle", 39: "Tech specialization", 40: "1st cycle degree", 
        42: "Prof. higher tech", 43: "Master (2nd cycle)"
    },
    "Nationality": {
        1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 14: "English",
        17: "Lithuanian", 21: "Angolan", 22: "Cape Verdean", 24: "Guinean", 25: "Mozambican",
        26: "Santomean", 32: "Turkish", 41: "Brazilian", 62: "Romanian", 100: "Moldova",
        101: "Mexican", 103: "Ukrainian", 105: "Russian", 108: "Cuban", 109: "Colombian"
    },
    "Displaced": {0: "No", 1: "Yes"},
    "Educational special needs": {0: "No", 1: "Yes"},
    "Debtor": {0: "No", 1: "Yes"},
    "Tuition fees up to date": {0: "No", 1: "Yes"},
    "Gender": {0: "Female", 1: "Male"},
    "Scholarship holder": {0: "No", 1: "Yes"},
    "International": {0: "No", 1: "Yes"},
}

# Sidebar inputs
st.sidebar.header("Input Mahasiswa")

def select_input(label, options_dict):
    reverse = {v: k for k, v in options_dict.items()}
    value = st.sidebar.selectbox(label, list(options_dict.values()))
    return reverse[value]

data = {}
for col in [
    "Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nationality", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"
]:
    data[col] = select_input(col, choices[col])

# Numeric inputs
data["Application order"] = st.sidebar.number_input("Application order (0-9)", 0, 9, 0)
data["Previous qualification (grade)"] = st.sidebar.slider("Previous qualification (grade)", 0, 200, 150)
data["Mother's qualification"] = st.sidebar.number_input("Mother's qualification (kode)", 1, 44, 1)
data["Father's qualification"] = st.sidebar.number_input("Father's qualification (kode)", 1, 44, 1)
data["Mother's occupation"] = st.sidebar.number_input("Mother's occupation (kode)", 0, 194, 0)
data["Father's occupation"] = st.sidebar.number_input("Father's occupation (kode)", 0, 195, 0)
data["Admission grade"] = st.sidebar.slider("Admission grade", 0, 200, 150)
data["Age at enrollment"] = st.sidebar.number_input("Age at enrollment", 16, 70, 18)
data["Curricular units 1st sem (credited)"] = st.sidebar.number_input("Credited units (1st sem)", 0, 20, 0)
data["Curricular units 1st sem (enrolled)"] = st.sidebar.number_input("Enrolled units (1st sem)", 0, 20, 5)
data["Curricular units 1st sem (evaluations)"] = st.sidebar.number_input("Evaluations (1st sem)", 0, 20, 3)
data["Curricular units 1st sem (approved)"] = st.sidebar.number_input("Approved units (1st sem)", 0, 20, 2)

# Convert to DataFrame
input_df = pd.DataFrame([data])

# Prediction
if st.button("Prediksi Status Mahasiswa"):
    prediction = model.predict(input_df)[0]
    label = label_mapping.get(prediction, "Unknown")
    st.subheader(f"Prediksi: {label}")
