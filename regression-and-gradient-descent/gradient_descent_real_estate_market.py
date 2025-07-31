import numpy as np

# Assume prices are linearly related to the size of the house
house_size = np.array([[1200], [1500], [1000]]) # Square feet
prices = np.array([240000, 300000, 200000]) # Price in dollars
theta = np.array([100000, 100]) # Initial guess for parameters [b, a]

# Perform one iteration of gradient descent to update the parameters
learning_rate = 0.01
m = len(prices)
X_b = np.c_[np.ones((3, 1)), house_size] # Adding bias term

# Calculate predictions
predictions = X_b.dot(theta)

# Calculate errors
errors = predictions - prices

# Calculate gradients
gradients = 2/m * X_b.T.dot(errors)

# Update parameters
theta = theta - learning_rate * gradients

print(f"Updated parameters: {theta}")
print(f"These parameters represent the linear model: price = {theta[0]:.2f} + {theta[1]:.2f} * size")