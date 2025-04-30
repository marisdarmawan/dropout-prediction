import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
model = joblib.load('model_pipeline.pkl')

# Load feature names from training (you can save this from training if needed)
# But for simplicity, infer from model directly
# You may also manually define it
feature_names = model.named_steps['feature_engineering'].func.__code__.co_varnames

st.title("Student Dropout Prediction")
st.subheader("Masukkan data mahasiswa:")

# Hardcode input features for safety and order
input_features = [
    'Age_at_enrollment', 'Previous_qualification_grade', 'Admission_grade',
    'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade',
    'Unemployment_rate', 'Inflation_rate', 'GDP',
    'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Mothers_qualification', 'Fathers_qualification',
    'Mothers_occupation', 'Fathers_occupation', 'Displaced',
    'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date',
    'Gender', 'Scholarship_holder', 'International'
]

input_data = {}
for feature in input_features:
    value = st.text_input(f"{feature}", "")
    input_data[feature] = value

if st.button("Predict Dropout"):
    try:
        input_df = pd.DataFrame([input_data])
        # Convert numeric columns (if any)
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except ValueError:
                pass  # keep as string for categorical

        # Predict
        prediction = model.predict(input_df)[0]

        # Output result
        labels = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
        st.success(f"The student is predicted to: **{labels[prediction]}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
