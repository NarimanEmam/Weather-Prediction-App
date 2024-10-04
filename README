# Weather Prediction ML Model
## Project Overview
This project aims to develop a predictive weather model using data from the Visual Crossing Weather API. The model will forecast temperature and classify weather conditions (e.g., sunny, rainy) based on various meteorological variables.
## Motivation
Accurate weather predictions are critical for sectors like agriculture, disaster management, transportation, and tourism. Leveraging multiple weather models improves accuracy and geographical specificity.
## Data Source
- Visual Crossing API: Provides historical and real-time weather data globally.
- Data Collected: Temperature, humidity, wind speed, precipitation, solar radiation, etc.
- Team Data Access: Each member collects 1000 rows of data per day, covering various geographical locations and time periods.
## Data Collection and Processing
- Data collected includes temperature, weather conditions, humidity, wind speed, and precipitation.
- Data Cleaning: Removal of irrelevant columns and outliers, handling of missing values.
- Feature Engineering: Extracted time-related features (day, month, year) to capture temporal weather patterns.
- Correlation Analysis: Identified strong relationships between weather variables, helping optimize the model.
## Model Training and Evaluation
The dataset was used to train several regression models:

1. Linear Regression: R² = 0.962, MSE = 4.63
2. K-Nearest Neighbors: R² = 0.963, MSE = 4.54
3. Support Vector Regression: R² = 0.687, MSE = 38.57
4. Decision Tree Regression: R² = 0.982, MSE = 2.12
5. Random Forest Regression: R² = 0.993, MSE = 0.84 (Best Performance)
## Streamlit Interface
An interactive Streamlit app allows users to input weather parameters and predict the temperature. The interface includes:
- Input fields for variables like humidity, wind speed, and solar radiation.
- A "Predict Temperature" button that uses a pre-trained Random Forest model to forecast temperature.
## How to Run
In the terminal after running the code, write this in terminal to open the streamlit page: 
```bash
    streamlit run app.py
```
