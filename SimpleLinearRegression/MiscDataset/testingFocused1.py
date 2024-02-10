import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# Load the dataset
df = pd.read_csv('linearRawData.csv')

# Drop unnecessary columns
df.drop('Unnamed: 0', axis=1, inplace=True)

# Define features and target
features = df[['feature1']]
target = df['target']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=5)

# Instantiate and fit the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate residuals
residuals = y_test - y_pred
residual_variance = np.var(residuals)

# Print the evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Median Absolute Error (MedAE): {medae}')
print(f'R^2 Score: {r2}')
print(f'Residual Variance: {residual_variance}')
