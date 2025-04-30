import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Ambil nama fitur dari model
feature_names = model.feature_names_in_

# Judul aplikasi
st.title("Student Dropout Prediction")
st.subheader("Input Data untuk Prediksi")

# Form input data
input_data = {}
for feature in feature_names:
    input_value = st.number_input(f"{feature}", format="%.4f")
    input_data[feature] = input_value

# Tombol prediksi
if st.button("Prediksi"):
    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        label_mapping = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
        st.success(f"Prediksi Status Mahasiswa: **{label_mapping[prediction]}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
