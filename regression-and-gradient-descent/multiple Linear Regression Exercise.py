
import numpy as np

# House features: [Size (sq ft), Number of rooms, Age (years)]
X = np.array([[2100, 3, 20], 
              [1600, 3, 15], 
              [2400, 4, 30], 
              [1416, 2, 20], 
              [3000, 5, 8]], dtype='float32')

# Prices
y = np.array([400000, 330000, 369000, 232000, 539900], dtype='float32')

# Add a column of 1's to the matrix X to account for the intercept
# This creates the design matrix for multiple linear regression
X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
print(f"Original X shape: {X.shape}")
print(f"X with intercept shape: {X_with_intercept.shape}")
print(f"X with intercept:\n{X_with_intercept}")

# Calculate the coefficients (beta) using the Normal Equation
beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

# Print the calculated beta coefficients
print(f"Beta coefficients:\n{beta}")