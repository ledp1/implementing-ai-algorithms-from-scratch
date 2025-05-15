import numpy as np
import matplotlib.pyplot as plt
X = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

y = np.array([56, 81, 119, 22, 103], dtype='float32')

# Add a column of ones to the X matrix to represent the intercept term
ones = np.ones(shape=(len(X), 1))
X = np.append(ones, X, axis=1)

# Calculate the beta coefficients using the normal equation
beta = np.linalg.inv(X.T @ X) @ X.T @ y # Shape of beta: (3, 1). Shape of X: (5, 3). Shape of y: (5, 1).

# Make predictions
predictions = X.dot(beta)

# Calculate the sum of squared residuals
ss_residuals = np.sum(np.square(y - predictions))

# Calculate the total sum of squares
ss_total = np.sum(np.square(y - np.mean(y)))

# Calculate the R^2 score
r2_score = 1 - (ss_residuals/ss_total)

# Print the R^2 score
print("R^2 Score:", r2_score) # Output: R^2 Score: 0.999187741755682