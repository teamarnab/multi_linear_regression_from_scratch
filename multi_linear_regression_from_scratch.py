# Import necessary libraries
import numpy as np
import pandas as pd

# For simplicity, we'll create a small synthetic dataset. Let's say we have a dataset with three features (x1, x2, x3) and a target variable (y)


# Step 1: Prepare the data
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [22, 25, 28, 30, 35, 40, 42, 45, 50, 55],
    'Education': [16, 17, 18, 18, 19, 20, 20, 21, 21, 22],
    'Salary': [20000, 25000, 30000, 35000, 40000, 50000, 55000, 60000, 65000, 70000]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)


# Step 2: Separate the features (X) and target variable (y)
X = df.drop(["Salary"], axis=1).values
y = df['Salary'].values


# Need to add a column of ones to X to account for the intercept term. This makes the matrix multiplication compatible with the intercept term

# Step 3: Add a column of ones to X to account for the intercept term

#The number of rows needed
rows_needed = X.shape[0]

#Creating a column of zeros and converting it into dataframe
ones = np.ones((rows_needed, 1)).astype(int)
ones

#Adding it to our Dataframe
X = np.append(ones, X, axis=1)


# Compute the coefficients (theta) using the Normal Equation. Formula: theta = (X.T * X)^(-1) * X.T * y

# Step 4:

#Transposing X matrix
X_transpose = X.T


#Finding the value of B0, B1, B2, B3
#(B0 is Bias)
#(B1, B2, B3 are the weights or coefficients of X1, X2 and X3 respectively)
betas = np.linalg.inv(X_transpose @ X) @ X_transpose @ y


# Print if needed
'''
print(f"Value of B0 (Bias) is: {betas[0]}")
print(f"Value of B1 (Weight for X1) is: {betas[1]}")
print(f"Value of B2 (Weight for X2) is: {betas[2]}")
print(f"Value of B3 (Weight for X3) is: {betas[3]}")
'''

# Creating a predicion function
def predict(new_arr):
  y_pred = betas[0] + (betas[1] * new_arr[0]) + (betas[2] * new_arr[1]) + (betas[3] * new_arr[2])
  return y_pred


# Predicting using prediction function
new_data = np.array([3,	28,	18]) #Data from 3rd row of our dataset
y_pred = predict(new_data)


# Finding error
y_true = 30000
error = y_true - y_pred
print(f"Error is: {error}")


# Assembling the code
class Regression:
  # Muliple linear regression algorihm
  def train_model(self, x_array, y_array):
    x_arr_b = np.c_[np.ones((x_array.shape[0], 1)).astype(int), x_array]
    betas = np.linalg.inv(x_arr_b.T @ x_arr_b) @ x_arr_b.T @ y_array
    return betas
  
  # Prediction functiom
  def predict(self, new_arr):
    y_pred = betas[0] + (betas[1] * new_arr[0]) + (betas[2] * new_arr[1]) + (betas[3] * new_arr[2])
    return y_pred
  
  # Scoring method
  def error(self, y_true, y_pred):
    error = y_true - y_pred
    error_score = error / y_true
    return error_score

linear_regression = Regression()

# Let's see if it works
test_x_arr = df.drop(["Salary"], axis=1).values
test_y_arr = df['Salary'].values
new_betas = linear_regression.train_model(test_x_arr, test_y_arr)


# Model is trained. Now predicting.
new_data_1 = np.array([10,	55,	22])
prediction = linear_regression.predict(new_data_1)
print(prediction)


# Checking error
y_actual = 70000
score = linear_regression.error(y_actual, prediction)
print(f"model's evaluation score is: {score}")