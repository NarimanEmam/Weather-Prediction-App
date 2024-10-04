import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

# Load the data
data2 = pd.read_csv('cleaned_data.csv')

# Label encode the 'conditions' categorical column
lab = LabelEncoder()
if 'conditions' in data2.columns:
    data2['conditions'] = lab.fit_transform(data2['conditions'])

# Define features and target
X = data2.drop(['temp'], axis=1)
y = data2['temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit GUI
st.title("Temperature Prediction Application")

# Inputs for weather conditions
st.subheader("Enter weather conditions to predict temperature:")

# Create input fields for the user
dew = st.number_input('Dew Point (°C)', value=9.5)
humidity = st.number_input('Humidity (%)', value=60.0)
windgust = st.number_input('Wind Gust (km/h)', value=30.0)
windspeed = st.number_input('Wind Speed (km/h)', value=15.0)
winddir = st.number_input('Wind Direction (°)', value=200.0)
sealevelpressure = st.number_input('Sea Level Pressure (hPa)', value=1015.0)
cloudcover = st.number_input('Cloud Cover (%)', value=10.0)
visibility = st.number_input('Visibility (km)', value=7.0)
solarradiation = st.number_input('Solar Radiation (W/m²)', value=165.0)
solarenergy = st.number_input('Solar Energy (kWh)', value=14.0)
uvindex = st.number_input('UV Index', value=5)
moonphase = st.number_input('Moon Phase', value=0.52)
conditions = st.selectbox('Conditions', ['Clear', 'Cloudy', 'Rainy', 'Snow', 'Fog'])
Day = st.number_input('Day', value=1)
Month = st.number_input('Month', value=1)
Year = st.number_input('Year', value=2024)

# Encode 'conditions' as per the label encoder used during training
conditions_mapping = {
    'Clear': 0,
    'Cloudy': 1,
    'Rainy': 2,
    'Snow': 3,
    'Fog': 4
}
conditions_encoded = conditions_mapping[conditions]

# Create the input dataframe in the exact order as the training data
input_data = pd.DataFrame([[dew, humidity, windgust, windspeed, winddir, sealevelpressure,
                            cloudcover, visibility, solarradiation, solarenergy, uvindex,
                            moonphase, conditions_encoded, Day, Month, Year]],
                          columns=X_train.columns)

# Predict button
if st.button('Predict Temperature'):
    # Make a prediction
    prediction = rf_model.predict(input_data)[0]
    
    # Display the result
    st.write(f"The predicted temperature is: {prediction:.2f} °C")
