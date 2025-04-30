st.title("Student Dropout Prediction")

st.subheader("Input Data untuk Prediksi")

# Dynamic form
input_data = {}
for feature in X.columns:
    input_value = st.number_input(f"Input {feature}", min_value=0.0)
    input_data[feature] = input_value

if st.button("Predict Dropout"):
    input_df = pd.DataFrame([input_data])
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Display the prediction
    if prediction == 0:
        st.write('The student is predicted to Dropout.')
    elif prediction == 1:
        st.write('The student is predicted to Graduate.')
    else:
        st.write('The student is predicted to be Enrolled.') 
