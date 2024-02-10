import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# Load the dataset
df = pd.read_csv('linearRawData.csv')

# Drop unnecessary columns and convert target to Series
target = df.drop('Unnamed: 0', axis=1).squeeze()

# Define features and split the dataset into training and testing sets
features = df[['feature1']]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=5)

# Instantiate and fit the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Calculate evaluation metrics
metrics = [mean_squared_error, mean_absolute_error, median_absolute_error, r2_score]
metric_names = ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'Median Absolute Error (MedAE)', 'R^2 Score']
metrics_values = [metric(y_test, y_pred) for metric in metrics]

# Calculate residuals and residual variance
residuals = y_test - y_pred
residual_variance = np.var(residuals, axis=0)

# Print the evaluation metrics
for name, value in zip(metric_names, metrics_values):
    print(f'{name}: {value}')

print(f'Residual Variance: {residual_variance}')