import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the CSV file as a pandas DataFrame
df = pd.read_csv('your_file.csv')

# Define the feature and target variables
X = df.drop('target_variable', axis=1)  # replace 'target_variable' with your actual target variable name
y = df['target_variable']

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize the linear regression model
reg = LinearRegression()

# Fit the model to the training data
reg.fit(X_train, y_train)

mse = np.mean((y_test - y_pred)**2)
residuals = y_test - y_pred

# Print the results
print("Model intercept: ", reg.intercept_)
print("Model coefficients: \n", reg.coef_)
print("\nMSE: ", mse)
print("Residuals: \n", residuals)
