import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('linearRawData.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

features = df[['feature1']]
target = df['target']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

residuals = y_test - y_pred
residual_variance = np.var(residuals)

print(f'Mean squared error: {mse}')
print(f'Residual: {residual_variance}')
print(f'r2 score: {r2}')