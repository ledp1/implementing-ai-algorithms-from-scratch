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