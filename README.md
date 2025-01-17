# PredictiveMaintainance 


# RUL Prediction App

# Overview

The RUL (Remaining Useful Life) Prediction App is a Streamlit-based web application designed to predict the remaining useful life of machinery based on input features. Users can either upload a CSV file with the necessary features or input the data manually.

# Features

# File Upload: Upload a CSV file containing the features necessary for RUL prediction.

# Manual Input: Manually input feature values for RUL prediction.

# Data Visualization: View the uploaded data and predicted RUL.

# Download Predictions: Download the file with predicted RUL as a CSV.

## Prerequisites

Streamlit: A web app framework for machine learning and data science.

Pandas: For data manipulation and analysis.

NumPy: For numerical computations.

Pickle: For loading pre-trained models and scalers.

Scikit-learn: For data preprocessing and machine learning models.






Choose an input method (Upload a file or Input Manually).

Follow the on-screen instructions to upload your file or enter data manually.

View the predictions and download the updated file if needed.





# Input Features

The app expects the following features:

Data_No

Differential_pressure

Flow_rate

Time

Dust_feed

Dust

Output

The predicted Remaining Useful Life (RUL) is displayed on the app.

A condition column indicating the machine's health status is added to the dataset.



Acknowledgements

Streamlit

Pandas

NumPy

Scikit-learn


