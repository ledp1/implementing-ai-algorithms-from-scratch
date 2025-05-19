import numpy as np

# given data
housing_data = np.array([[1800, 3], [2400, 4], [1416, 2], [3000, 5]])
prices = np.array([350000, 475000, 230000, 640000])

# adding 1s to our matrix
ones = np.ones(shape=(len(housing_data), 1))
X = np.append(ones, housing_data, axis=1)

# calculating coefficients
coefficients = np.linalg.inv(X.T @ X) @ X.T @ prices

# predicting prices
predicted_prices = X @ coefficients

# calculating residuals 
residuals = prices - predicted_prices

# calculating total sum of squares 
sst = np.sum((prices - np.mean(prices)) ** 2)

# calculating residual sum of squares 
ssr = np.sum(residuals ** 2)

# calculating R^2
r2 = 1 - (ssr/sst)

print("Coefficients:", coefficients)
print("Predicted prices:", predicted_prices)
print("R^2:", r2)

# Results Analysis:
# 1. Coefficients:
#    - First value (-91666.67): Intercept term
#    - Second value (146.60): Coefficient for square footage
#    - Third value (57037.04): Coefficient for number of bedrooms
#
# 2. Predicted Prices vs Actual Prices:
#    - House 1: $343,333.33 (actual: $350,000) - Error: $6,666.67
#    - House 2: $488,333.33 (actual: $475,000) - Error: $13,333.33
#    - House 3: $230,000.00 (actual: $230,000) - Error: $0.00
#    - House 4: $633,333.33 (actual: $640,000) - Error: $6,666.67
#
# 3. R-squared (R²) = 0.9971 or 99.71%
#    - Indicates the model explains 99.71% of the variance in house prices
#    - Very high R² value suggests excellent model fit