import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


warnings.filterwarnings('ignore')

# List input files (if applicable)
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
       # print(os.path.join(dirname, filename))

# Load data
coffee_data = pd.read_csv('E:\\AICETE & Edunet\\Coffee Sales project details\\coffee_sales.csv')

# Data cleaning
coffee_data['date'] = pd.to_datetime(coffee_data['date'])
coffee_data['datetime'] = pd.to_datetime(coffee_data['datetime'])
coffee_data['month'] = coffee_data['date'].dt.strftime('%Y-%m')
coffee_data['day'] = coffee_data['date'].dt.strftime('%w')
coffee_data['hour'] = coffee_data['datetime'].dt.strftime('%H')

# Revenue by product
revenue_data = coffee_data.groupby('coffee_name', as_index=False)['money'].sum().sort_values(by='money', ascending=False)

plt.figure(figsize=(10, 4))
ax = sns.barplot(data=revenue_data, x='money', y='coffee_name', color='steelblue')
ax.bar_label(ax.containers[0], fontsize=6)
plt.xlabel('Revenue')
plt.title('Revenue by Coffee Product')
plt.tight_layout()
plt.savefig('revenue_by_product.png')
plt.close()

# Monthly sales per coffee type
monthly_sales = (
    coffee_data.groupby(['coffee_name', 'month'])['date']
    .count()
    .reset_index()
    .rename(columns={'date': 'count'})
    .pivot(index='month', columns='coffee_name', values='count')
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales)
plt.legend(loc='upper left')
plt.xticks(range(len(monthly_sales['month'])), monthly_sales['month'], size='small')
plt.title('Monthly Sales Trends by Coffee Type')
plt.tight_layout()
plt.show()
plt.savefig('monthly_sales_trends.png')
plt.close()

# Weekday sales
weekday_sales = coffee_data.groupby('day')['date'].count().reset_index().rename(columns={'date': 'count'})

plt.figure(figsize=(12, 6))
sns.barplot(data=weekday_sales, x='day', y='count', color='steelblue')
plt.xticks(range(7), ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], size='small')
plt.title('Sales by Weekday')
plt.tight_layout()
plt.show()
plt.savefig('weekday_sales.png')
plt.close()

# Hourly sales
hourly_sales = coffee_data.groupby('hour')['date'].count().reset_index().rename(columns={'date': 'count'})

plt.figure(figsize=(12, 6))
sns.barplot(data=hourly_sales, x='hour', y='count', color='steelblue')
plt.title('Sales by Hour')
plt.tight_layout()
plt.show()
plt.savefig('hourly_sales.png')
plt.close()

# Hourly sales by coffee product
hourly_sales_by_coffee = (
    coffee_data.groupby(['hour', 'coffee_name'])['date']
    .count()
    .reset_index()
    .rename(columns={'date': 'count'})
    .pivot(index='hour', columns='coffee_name', values='count')
    .fillna(0)
    .reset_index()
)

# Plot each product's hourly sales
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

for i, column in enumerate(hourly_sales_by_coffee.columns[1:]):
    axs[i].bar(hourly_sales_by_coffee['hour'], hourly_sales_by_coffee[column])
    axs[i].set_title(column)
    axs[i].set_xlabel('Hour')

plt.tight_layout()
plt.show()
plt.savefig('hourly_sales_by_product.png')
plt.close()


# Prepare dataset for ML

# Select features (exclude 'datetime', 'date', 'card' for now)
features = ['month', 'day', 'hour', 'cash_type', 'coffee_name']

# One-hot encode categorical features
X = pd.get_dummies(coffee_data[features], drop_first=True)

# Target variable
y = coffee_data['money']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\nMean Squared Error: {mse:.2f}\nR² Score: {r2:.2f}")

# Coefficients for interpretation
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nModel Coefficients:")
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Save model coefficients
coefficients.to_csv('model_coefficients.csv', index=False)