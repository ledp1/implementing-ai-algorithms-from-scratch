import numpy as np

# Constants for area (m^2), number of bedrooms, and age of the house (years)
features = np.array([[120, 3, 10],
                     [150, 4, 15],
                     [90, 2, 20],
                     [110, 3, 5]], dtype='float32')

# Corresponding house price in thousands of dollars
prices = np.array([[300],
                   [400],
                   [250],
                   [275]], dtype='float32')

# Adding a column of ones for the intercept term
ones = np.ones(shape=(len(features), 1))
X = np.append(ones, features, axis=1)



#TODO:  Calculate coefficients for the regression equation
coefficients = np.linalg.inv(X.T @ X) @ X.T @ prices

# Calculate predicted prices
predicted_prices = X.dot(coefficients)

# Calculate the total sum of squares
tss = ((prices - prices.mean())**2).sum()

# Calculate the residual sum of squares
rss = ((prices - predicted_prices)**2).sum()

# Calculate the R^2 score
r2 = 1 - rss/tss

print('Regression Coefficients Unflattened:', coefficients)
print('Regression Coefficients:', coefficients.flatten())
print('Predicted prices Unflattened:', predicted_prices)
print('Predicted prices:', predicted_prices.flatten())
print('R^2:', r2)