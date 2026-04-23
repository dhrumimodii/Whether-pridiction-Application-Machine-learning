# ============================================================
# WEATHER PREDICTION PROJECT (EASY VERSION)
# Linear Regression + Random Forest Classifier
# ============================================================


# -------------------------------
# STEP 1: IMPORT LIBRARIES
# -------------------------------
print("Step 1: Importing libraries...\n")

import pandas as pd                 # for data handling
import numpy as np                  # for numerical operations
import matplotlib.pyplot as plt     # for graphs
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

print("Libraries imported successfully!\n")


# -------------------------------
# STEP 2: LOAD DATASET
# -------------------------------
print("Step 2: Loading dataset...\n")

df = pd.read_csv("weather.csv")     # load dataset

print("Dataset loaded!\n")
print(df.head(), "\n")              # show first 5 rows


# -------------------------------
# STEP 3: DATA CLEANING
# -------------------------------
print("Step 3: Checking missing values...\n")

print(df.isnull().sum(), "\n")      # check missing values

df = df.dropna()                   # remove missing values

print("Missing values removed!\n")


# -------------------------------
# STEP 4: PREPARE DATA
# -------------------------------
print("Step 4: Preparing data...\n")

# Convert RainTomorrow Yes/No → 1/0
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})

print(df[['RainTomorrow']].head(), "\n")


# ============================================================
# 🔹 PART 1: LINEAR REGRESSION (Temperature Prediction)
# ============================================================

print("Step 5: Linear Regression...\n")

# Input features
X_reg = df[['MinTemp', 'MaxTemp', 'WindGustSpeed', 'Humidity', 'Pressure']]

# Target (Temperature)
y_reg = df['Temp']

print("Features:\n", X_reg.head())
print("Target:\n", y_reg.head(), "\n")


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape, "\n")


# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Linear Regression model trained!\n")


# Prediction
lr_pred = lr_model.predict(X_test)

print("First 5 Predictions:\n", lr_pred[:5])
print("Actual values:\n", y_test.values[:5], "\n")


# Error calculation
lr_error = mean_absolute_error(y_test, lr_pred)
print("Linear Regression Error:", lr_error, "\n")


# 📊 Graph for Linear Regression
plt.figure()

plt.plot(y_test.values[:50], marker='o', label="Actual")
plt.plot(lr_pred[:50], marker='x', label="Predicted")

plt.title("Linear Regression (Temperature)")
plt.xlabel("Data Points")
plt.ylabel("Temperature")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 🔹 PART 2: RANDOM FOREST CLASSIFIER (Rain Prediction)
# ============================================================

print("Step 6: Random Forest Classifier...\n")

# Input features (same)
X_clf = df[['MinTemp', 'MaxTemp', 'WindGustSpeed', 'Humidity', 'Pressure']]

# Target (RainTomorrow)
y_clf = df['RainTomorrow']


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape, "\n")


# Train model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

print("Random Forest model trained!\n")


# Prediction
rf_pred = rf_model.predict(X_test)

print("First 10 Predictions:\n", rf_pred[:10])
print("Actual values:\n", y_test.values[:10], "\n")


# Accuracy
rf_acc = accuracy_score(y_test, rf_pred)
print("Accuracy:", rf_acc, "\n")


# 📊 Graph for Classification
plt.figure()

plt.plot(y_test.values[:50], marker='o', label="Actual")
plt.plot(rf_pred[:50], marker='s', label="Predicted")

plt.title("Random Forest (Rain Prediction)")
plt.xlabel("Data Points")
plt.ylabel("Rain (1=Yes, 0=No)")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# STEP 7: SAMPLE PREDICTION
# ============================================================

print("Step 7: Sample Prediction...\n")

sample = [[10, 25, 40, 60, 1015]]

# Temperature prediction
temp_pred = lr_model.predict(sample)

# Rain prediction
rain_pred = rf_model.predict(sample)

print("Predicted Temperature:", temp_pred)

if rain_pred[0] == 1:
    print("Rain Tomorrow: Yes 🌧️")
else:
    print("Rain Tomorrow: No ☀️")

# Save both models
pickle.dump(lr_model, open("temp_model.pkl", "wb"))
pickle.dump(rf_model, open("rain_model.pkl", "wb"))

print("Models saved successfully!")

# -------------------------------
# Load Models
# -------------------------------
temp_model = pickle.load(open("temp_model.pkl", "rb"))
rain_model = pickle.load(open("rain_model.pkl", "rb"))

# -------------------------------
# UI Design
# -------------------------------
st.title("🌦 Weather Prediction App")

st.write("Enter weather details below:")

# Input fields
min_temp = st.number_input("Min Temperature")
max_temp = st.number_input("Max Temperature")
wind_speed = st.number_input("Wind Gust Speed")
humidity = st.number_input("Humidity")
pressure = st.number_input("Pressure")

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):

    input_data = np.array([[min_temp, max_temp, wind_speed, humidity, pressure]])

    # Predictions
    temp = temp_model.predict(input_data)
    rain = rain_model.predict(input_data)

    # Output
    st.subheader("Results:")

    st.success(f"🌡 Predicted Temperature: {temp[0]}")

    if rain[0] == 1:
        st.error("🌧 Rain Tomorrow: YES")
    else:
        st.success("☀️ Rain Tomorrow: NO")