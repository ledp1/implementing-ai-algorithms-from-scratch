import numpy as np

# Real estate prices (in $1000's) based on the size of houses (in 1000 sqft)
house_sizes = np.array([[1], [2], [3]])
house_prices = np.array([[300], [550], [760]])

# Initialize parameters for gradient descent
theta = np.array([[0.0], [1.0]])  # theta[0] is the intercept(b), theta[1] is the slope(a)
learning_rate = 0.01  # Initial learning rate
iterations = 10

# Prepend 1's to house_sizes to accommodate the intercept term (bias)
X_b = np.c_[np.ones((len(house_sizes), 1)), house_sizes]

# Perform gradient descent
for i in range(iterations):
    gradients = 2/len(house_prices) * X_b.T.dot(X_b.dot(theta) - house_prices)
    theta -= learning_rate * gradients

# Calculate predictions using the trained parameters
pred = X_b.dot(theta)

# Calculate R-squared
ssr = np.sum((pred - house_prices)**2)
sst = np.sum((house_prices - np.mean(house_prices))**2)
r2 = 1 - (ssr/sst)

print(f'Trained parameters: {theta.ravel()}, R-squared: {r2}')