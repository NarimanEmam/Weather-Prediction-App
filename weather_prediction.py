# -*- coding: utf-8 -*-
"""Weather_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dSyAqPM5Fa6JZew_lXJ-Vd9y_TAfLtDA

#Data Loading and Exploration
"""

import pandas as pd
import numpy as np
import  seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

from google.colab import drive
import pandas as pd


file_path = "/content/Egypt 2010-01-01 to 2024-09-12.csv"
data = pd.read_csv(file_path)

from google.colab import drive
drive.mount('/content/drive')

data.head()

data.shape

data.info()

data.describe()

data.isna().sum()

data.sample(5)

data.columns

"""#Data Cleaning

###a. Remove unnecessary columns
"""

data=data.drop(['Unnamed: 0'],axis=1)

data=data.drop(['severerisk'],axis=1)

data= data.drop(['name'] ,axis=1 )

data= data.drop(['snow'],axis=1)

data = data.drop(['snowdepth'],axis=1)

data = data.drop(['preciptype'],axis=1)

"""###b. Handle missing values"""

data['windgust'].isna().sum()

data['windgust']= data['windgust'].fillna(data['windgust'].median())

data.isna().sum()

data.dropna(subset=['visibility'], inplace=True)

data.shape

"""###c. Check for duplicates"""

data = data.drop_duplicates()

data.shape

"""#Convert Data Types"""

# Convert 'datetime' column to actual datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

data['sunset'] = pd.to_datetime(data['sunset'])

data['sunrise'] = pd.to_datetime(data['sunrise'])

data.info()

num_columns = data.describe().columns
num_rows = int(np.ceil(len(num_columns) / 4))  # Using 4 columns per row
plt.figure(figsize=(20, num_rows * 5))
for i, column in enumerate(num_columns):
    plt.subplot(num_rows, 4, i + 1)
    sns.boxplot(x=data[column])
    plt.title(column)
plt.tight_layout()
plt.show()

min_values = data.min()
max_values = data.max()
df = pd.DataFrame({'Min': min_values, 'Max': max_values})
print(df)

"""#Handling Outliers"""

# IQR method to clamp outliers
def outliers_clamping(col, value):
    Q1 = np.percentile(data[col], q = 25, interpolation = 'midpoint')
    Q3 = np.percentile(data[col], q = 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    Upper_Bound = Q3 + 1.5*IQR
    Lower_Bound = Q1 - 1.5*IQR
    if value > Upper_Bound:
        return Upper_Bound
    elif value<Lower_Bound:
        return Lower_Bound
    else:
        return value

num_cols=data.describe().columns
for col in num_cols:
  data[col] = data[col].apply(lambda x: outliers_clamping(col, x))

min_values = data.min()
max_values = data.max()
df = pd.DataFrame({'Min': min_values, 'Max': max_values})
print(df)

corr= data.select_dtypes(exclude=['object']).corr()

plt.figure(figsize=(12, 10))
plt.xticks(rotation=90)
sns.heatmap(corr, annot= True,cmap='coolwarm', annot_kws={'size': 8})
plt.show()

'''
White (or light colors) usually represent correlations close to zero, meaning there is little to no linear relationship between the variables.
'''

data.columns

data.columns

print(data['precipcover'].describe())

data['precipcover'].head(30)

data['precipcover'].unique()

data.drop(['precipcover'],axis=1,inplace=True)

data['precipprob'].unique()

data.drop(['precipprob'],axis=1,inplace=True)

data['precip'].unique()

data.drop(['precip'],axis=1,inplace=True)

data.drop(['tempmax','tempmin'], axis=1, inplace = True)

data.drop(['feelslikemin','feelslike','feelslikemax'], axis= 1 , inplace=True)

data['conditions'].unique()

data['icon'].unique()

data['description'].unique()

data.drop(['description'],axis=1,inplace=True)

data['stations'].unique()

data.drop(['stations'],axis=1,inplace=True)

data['Day'] =data['datetime'].dt.day
data['Month'] = data['datetime'].dt.month
data['Year'] = data['datetime'].dt.year

data.head()

corr= data.select_dtypes(exclude=['object']).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot= True,cmap='coolwarm', annot_kws={'size': 8})
plt.show()

"""##Visualizing Feature Distributions"""

data.hist(figsize=(20, 15))
plt.show()

"""
##Outlier Detection and Visualization Using Boxplots"""

num_columns = data.describe().columns
num_rows = int(np.ceil(len(num_columns) / 4))  # Using 4 columns per row
plt.figure(figsize=(20, num_rows * 5))
for i, column in enumerate(num_columns):
    plt.subplot(num_rows, 4, i + 1)
    sns.boxplot(x=data[column])
    plt.title(column)
plt.tight_layout()
plt.show()

data.shape

"""# Data Analysis Process

###Distribution of icon Type
"""

sns.countplot(x='icon', data=data , palette = 'Set2')
plt.title('Distribution of icon Type')
plt.xlabel('icon Type')
plt.show()

"""###Grouping the features of interest by month"""

monthly_data = data.groupby((data['Month']))[[
    'temp',
    'humidity',
    'windspeed',
    'winddir',
    'sealevelpressure',
    'visibility'
]]
# Creating subplot layout
fig, ax = plt.subplots(2,3, figsize=(15,10))

# Main figure title
fig.suptitle('Year-over-Year Changes in Recorded Features')
# Temperature line plot
sns.lineplot(x='Month', y='temp', data=data, color='tab:blue', ax=ax[0,0])
ax[0,0].set_ylabel('Temperature (C)')
ax[0,0].set_xlabel('Month')

# Humidity line plot
sns.lineplot(x='Month', y='humidity', data=data, color='tab:red', ax=ax[0,1])
ax[0,1].set_ylabel('Humidity (%)')
ax[0,1].set_xlabel('Month')

# Wind speed line plot
sns.lineplot(x='Month', y='windspeed', data=data, color='tab:purple', ax=ax[0,2])
ax[0,2].set_ylabel('Wind Speed (km/h)')
ax[0,2].set_xlabel('Month')

# Wind bearing line plot
sns.lineplot(x='Month', y='winddir', data=data, color='tab:green', ax=ax[1,0])
ax[1,0].set_ylabel('Wind Direction')
ax[1,0].set_xlabel('Month')

# Visibility line plot
sns.lineplot(x='Month', y='visibility', data=data, color='tab:pink', ax=ax[1,1])
ax[1,1].set_ylabel('Visibility (km)')
ax[1,1].set_xlabel('Month')

# Pressure line plot
sns.lineplot(x='Month', y='sealevelpressure', data=data, color='tab:orange', ax=ax[1,2])
ax[1,2].set_ylabel('Pressure (milibars)')
ax[1,2].set_xlabel('Month')

"""### Grouping the features of interest by year"""

monthly_data = data.groupby((data['Year']))[[
    'temp',
    'humidity',
    'windspeed',
    'winddir',
    'sealevelpressure',
    'visibility'
]]

# Creating subplot layout
fig, ax = plt.subplots(2,3, figsize=(15,10))

# Main figure title
fig.suptitle('Year-over-Year Changes in Recorded Features')
# Temperature line plot
sns.lineplot(x='Year', y='temp', data=data, color='tab:blue', ax=ax[0,0])
ax[0,0].set_ylabel('Temperature (C)')
ax[0,0].set_xlabel('Year')

# Humidity line plot
sns.lineplot(x='Year', y='humidity', data=data, color='tab:red', ax=ax[0,1])
ax[0,1].set_ylabel('Humidity (%)')
ax[0,1].set_xlabel('Year')

# Wind speed line plot
sns.lineplot(x='Year', y='windspeed', data=data, color='tab:purple', ax=ax[0,2])
ax[0,2].set_ylabel('Wind Speed (km/h)')
ax[0,2].set_xlabel('Year')

# Wind bearing line plot
sns.lineplot(x='Year', y='winddir', data=data, color='tab:green', ax=ax[1,0])
ax[1,0].set_ylabel('Wind Direction')
ax[1,0].set_xlabel('Year')

# Visibility line plot
sns.lineplot(x='Year', y='visibility', data=data, color='tab:pink', ax=ax[1,1])
ax[1,1].set_ylabel('Visibility (km)')
ax[1,1].set_xlabel('Year')

# Pressure line plot
sns.lineplot(x='Year', y='sealevelpressure', data=data, color='tab:orange', ax=ax[1,2])
ax[1,2].set_ylabel('Pressure (milibars)')
ax[1,2].set_xlabel('Year')

"""### insight
Pressure: High Air Pressure (or high-pressure systems): These usually bring clear skies, calm weather, and dry conditions.
Low Air Pressure (or low-pressure systems): These often lead to cloudy, windy, and rainy conditions, sometimes resulting in storms or more turbulent weather.

###Frequently Temperature
"""

plt.figure(figsize=(8, 6))
sns.histplot(data=data['temp'], bins=20, kde=True , color = 'green')
plt.title('Distribution of Temperature')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()

"""

```
# This is formatted as code
```

###Condition Frequency"""

n = len(data['conditions'].unique())
con = data['conditions'].value_counts().head(n)
color = (0.2, # redness
         0.4, # greenness
         0.2, # blueness
         0.6 # transparency
         )
plt.figure(figsize=(10, 6))
sns.barplot(x=con.values, y=con.index , color = color)
plt.title(f' {n} Condition Frequency')
plt.xlabel('Frequency')
plt.ylabel('Conditions')
plt.tight_layout()
plt.show()

"""###Plot temperature over time

"""

plt.figure(figsize=(10, 6))
sns.lineplot(x=data['datetime'], y=data['temp'], color='blue')
plt.title('Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.show()

"""### Temperature vs. Humidity

"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['temp'], y=data['humidity'], hue=data['conditions'], palette='coolwarm')
plt.title('Temperature vs Humidity with Weather Conditions')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.tight_layout()
plt.show()

"""## Insight:
This graph helps visualize the relationship between temperature and humidity, colored by weather conditions. For example, clear days might have low humidity, while cloudy days could have higher humidity levels.

### Wind Speed and Wind Gust Analysis
"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['windspeed'], y=data['windgust'], hue=data['conditions'], palette='Spectral')
plt.title('Wind Speed vs Wind Gust')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Wind Gust (km/h)')
plt.tight_layout()
plt.show()

"""## Insight:
This plot can reveal how gusty certain weather conditions are. For instance, clear or calm days might have lower gusts compared to stormy or windy conditions.

###  Cloud Cover vs. Solar Radiation
"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['cloudcover'], y=data['solarradiation'], hue=data['conditions'], palette='viridis')
plt.title('Cloud Cover vs Solar Radiation')
plt.xlabel('Cloud Cover (%)')
plt.ylabel('Solar Radiation (W/m²)')
plt.tight_layout()
plt.show()

"""## Insight:
This examines how cloud cover impacts solar radiation. Days with high cloud cover typically receive less solar radiation.

This relationship is critical for energy sectors, especially for predicting solar energy output on cloudy days.

### Visibility and Conditions
"""

plt.figure(figsize=(10, 6))
sns.boxplot(x=data['conditions'], y=data['visibility'])
plt.title('Visibility Across Different Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Visibility (km)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

"""## Insight:
This plot helps to understand how different weather conditions impact visibility. For example, clear days usually have higher visibility, while foggy or rainy conditions reduce visibility.

### Solar Energy by Month
"""

# Grouping data by month to get average solar energy for each month
monthly_avg = data.groupby('Month')['solarenergy'].mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
plt.title('Average Solar Energy by Month')
plt.xlabel('Month')
plt.ylabel('Average Solar Energy (kWh/m²)')
plt.grid(True)
plt.tight_layout()
plt.show()

"""## Insight:
This boxplot shows how solar energy changes throughout the year. You may observe higher energy in summer months ( June to August) and lower energy during winter months ( December to February)

###  Wind Direction Distribution
"""

plt.figure(figsize=(10, 6))
sns.histplot(data['winddir'], bins=36, color='orange')
plt.title('Wind Direction Distribution')
plt.xlabel('Wind Direction (°)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

"""## Insight:
This histogram shows the frequency of different wind directions, which can help determine prevailing winds in the area (e.g., most wind might come from the west or northwest).

### UV Index and Conditions
"""

plt.figure(figsize=(10, 6))
sns.boxplot(x=data['conditions'], y=data['uvindex'])
plt.title('UV Index by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('UV Index')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

"""## Insight:
 Clear days are likely to have a higher UV index compared to cloudy or overcast days, indicating more exposure to ultraviolet rays.

### Temperature, Sunrise, and Sunset Timing
"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pd.to_datetime(data['sunrise']).dt.hour, y=data['temp'], color='purple', label='Sunrise')
sns.scatterplot(x=pd.to_datetime(data['sunset']).dt.hour, y=data['temp'], color='orange', label='Sunset')
plt.title('Temperature at Sunrise and Sunset Times')
plt.xlabel('Hour of Day')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.legend()
plt.show()

"""## Insight:
This visualization shows how temperature varies throughout the day relative to sunrise and sunset times. You might notice how temperatures start rising after sunrise and fall after sunset.
"""

data.drop(['datetime'],axis=1,inplace=True)

data.sample(10)

data.head()

data.to_csv('cleaned_data.csv', index=False)

data2 = pd.read_csv('cleaned_data.csv')

data2.drop(['icon'],axis=1,inplace=True)

obj=data2.select_dtypes(include="object")
num=data2.select_dtypes(exclude="object")
lab = LabelEncoder()
obj = obj.apply(lambda col: lab.fit_transform(col) if col.dtype == 'object' else col)
data2=pd.concat([obj,num],axis=1)

data2.dtypes

data2.info()

"""#Modeling"""

X = data2.drop(['temp'], axis=1)
y = data2['temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""##Linear Regression"""

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
#r2_adj_lr = 1 - (1 - r2_lr) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
mse_lr = mean_squared_error(y_test, y_pred_lr)


print(f'Linear Regression - R² Score: {r2_lr:.5f}')
#print(f'Linear Regression - Adjusted R² Score: {r2_adj_lr:.5f}')
print(f'Linear Regression - Mean Squared Error: {mse_lr:.5f}')

"""##KNN"""

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

r2_knn = r2_score(y_test, y_pred_knn)
#r2_adj_knn = 1 - (1 - r2_knn) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
mse_knn = mean_squared_error(y_test, y_pred_knn)

print(f'KNN - R² Score: {r2_knn:.5f}')
#print(f'KNN - Adjusted R² Score: {r2_adj_knn:.5f}')
print(f'KNN - Mean Squared Error: {mse_knn:.5f}')

"""##SVR"""

svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)

r2_svr = r2_score(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)

print(f'SVR - R² Score: {r2_svr:.5f}')
print(f'SVR - Mean Squared Error: {mse_svr:.5f}')

"""##Desicion Tree"""

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

r2_dt = r2_score(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)

print(f'Decision Tree Regression - R² Score: {r2_dt:.5f}')
print(f'Decision Tree Regression - Mean Squared Error: {mse_dt:.5f}')

"""##Random Forest Regressor"""

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
#r2_adj_rf = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f'Random Forest Regression - R² Score: {r2_rf:.5f}')
#print(f'Random Forest Regression - Adjusted R² Score: {r2_adj_rf:.5f}')
print(f'Random Forest Regression - Mean Squared Error: {mse_rf:.5f}')

"""#Models Comparison"""

results = {
    'Model': ['Linear Regression', 'KNN', 'SVR', 'Decision Tree','Random Forest'],
    'R² Score': [r2_lr, r2_knn, r2_svr, r2_dt,r2_rf],
    'Mean Squared Error': [mse_lr, mse_knn, mse_svr, mse_dt, mse_rf]
}

results_df = pd.DataFrame(results)

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.xticks(rotation=45)
plt.title('R² Score Comparison')

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Mean Squared Error', data=results_df, palette='viridis')
plt.xticks(rotation=45)
plt.title('Mean Squared Error Comparison')

plt.tight_layout()
plt.show()