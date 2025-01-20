import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Loading the model and scalers
MODEL_PATH = "random_forest_model.pkl"
FEATURE_SCALER_PATH = "feature_scaler.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(FEATURE_SCALER_PATH, "rb") as feature_scaler_file:
    feature_scaler = pickle.load(feature_scaler_file)

with open(TARGET_SCALER_PATH, "rb") as target_scaler_file:
    target_scaler = pickle.load(target_scaler_file)

# App title and description
st.title("RUL Prediction App")
st.write("""The features of the file should be:
                \n Data_No,\n Differential_pressure,\n Flow_rate,\n Time,\n Dust_feed,\n Dust
""")

# Loading the CSS file
with open("style.css") as f:
    css = f.read()


st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.logo("Background_img.jpg")

Input_method = st.sidebar.selectbox("Choose Your Option", ["Choose","Upload a file", "Input Manually"])


if Input_method =="Upload a file":
# File uploader
    st.sidebar.write("Upload a CSV file with features to predict the Remaining Useful Life (RUL).")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Reading the uploaded file
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Your uploaded file is :")
            st.write(data.head())
            
            # Check if the required features are present
            missing_features = set(feature_scaler.feature_names_in_) - set(data.columns)
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
            else:
                #label encoding
                label_enc = LabelEncoder()
                data["Dust"] = label_enc.fit_transform(data["Dust"])
                
                # Scale the features
                features_scaled = feature_scaler.transform(data)

                # Predict the scaled target
                y_scaled_prediction = model.predict(features_scaled)

                # Inverse transform the predictions to the original scale
                y_original_prediction = target_scaler.inverse_transform(y_scaled_prediction.reshape(-1, 1))
                #to get prediction in range of 0 - 100
                y_original_prediction = np.clip(y_original_prediction, 0, 100)

                # Append predictions to the original DataFrame
                data["Predicted_RUL"] = y_original_prediction
                
                #Adding Condition Column
                def condition(rul):
                    if 15 < rul < 30:
                        return "The Machine needs some care"
                    elif rul <= 15:
                        return "The Machine needs critical care"
                    else:
                        return "The Machine is in good condition"
                    
                data["condition"] = data["Predicted_RUL"].apply(condition)

                st.write("File with Predicted RUL Column is :")
                st.write(data.head())

                # Download the updated file
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predicted_rul.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
elif Input_method == "Input Manually":
    st.sidebar.write("Manually input the features to predict RUL:")
    
    col1, col2 = st.columns(2)
    with col1:
        data_no = st.number_input("Data_No", min_value=0, max_value=50)
        if data_no > 50 or data_no < 0:
            st.error("Data_No should be between 0 and 50")
        differential_pressure = st.number_input("Differential_pressure", min_value=0.0, value=0.0)
        flow_rate = st.number_input("Flow_rate", min_value=0.0, value=0.0)
    with col2:
        time = st.number_input("Time", min_value=0.0, value=0.0)
        dust_feed = st.number_input("Dust_feed", min_value=0.0, value=0.0)
        dust = st.selectbox("Dust", options=["A3 Medium Test Dust", " A2 Fine Test Dust", "A4 Coarse Test Dust"])
        
    
    
    if st.button("Predict RUL"):
        label_enc = LabelEncoder()
        label_enc.fit(["A3 Medium Test Dust", " A2 Fine Test Dust", "A4 Coarse Test Dust"])
        dust_encoded = label_enc.transform([dust])[0]

        manual_data = pd.DataFrame({
            'Data_No': [data_no],
            'Differential_pressure': [differential_pressure],
            'Flow_rate': [flow_rate],
            'Time': [time],
            'Dust_feed': [dust_feed],
            'Dust': [dust_encoded]
        })

        features_scaled_manual = feature_scaler.transform(manual_data)
        y_scaled_prediction_manual = model.predict(features_scaled_manual)
        y_original_prediction_manual = target_scaler.inverse_transform(y_scaled_prediction_manual.reshape(-1, 1))
        
        y_original_prediction_manual = np.clip(y_original_prediction_manual, 0, 100)
        

        st.write(f"Predicted RUL: {y_original_prediction_manual[0][0]:.2f}")
        
        if y_original_prediction_manual < 30 and y_original_prediction_manual > 15:
            st.write("The Machine needs some care")
        elif y_original_prediction_manual < 15:
            st.write("The Machine needs critical care")
        else:
            st.write("The Machine is in good condition")
        