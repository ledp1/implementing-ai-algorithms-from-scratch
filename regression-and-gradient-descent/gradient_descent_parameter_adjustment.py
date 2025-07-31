import numpy as np

# Sample house sizes in square feet, standardized
house_sizes = np.array([[1000], [1500], [2000]])
house_sizes = (house_sizes - np.mean(house_sizes)) / np.std(house_sizes)
# Sample house prices in 1000s of dollars
house_prices = np.array([[300], [450], [600]])
# We initialize our parameters: slope (a) and intercept (b)
theta_real_estate = np.random.rand(2, 1)
# Learning rate and iterations for gradient descent, adjusted learning rate
alpha_real_estate = 0.01
iterations = 500
# Add a column of ones to the house sizes to accommodate the intercept (b)
X_b_real_estate = np.c_[np.ones((len(house_sizes), 1)), house_sizes]

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):  # Iterate until convergence
        prediction = np.dot(X, theta)  # Matrix multiplication between X and theta
        
        # Calculate gradient
        error = prediction - y
        gradient = (1/m) * np.dot(X.T, error)
        
        # Update parameters
        theta = theta - alpha * gradient
        
        # Calculate and record cost
        cost = np.sum((prediction - y) ** 2) / (2 * m)
        cost_history[i] = cost
        
    return theta, cost_history

# Run gradient descent to update parameters and get cost history
theta_real_estate, cost_history = gradient_descent(X_b_real_estate, house_prices, theta_real_estate, alpha_real_estate, iterations)

for i, cost in enumerate(cost_history[::10]):
    print(f'Iteration {i * 10}: Cost = {cost}')