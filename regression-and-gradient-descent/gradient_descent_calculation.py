def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for i in range(iterations): # Iterate until convergence
        prediction = np.dot(X,theta)  # Matrix multiplication between X and theta
        theta = theta - (1/m)*alpha*(X.T.dot((prediction - y))) # Gradient update rule
        theta_history[i,:] = theta.T
        cost_history[i] = cost(X,y,theta)
    return theta, cost_history, theta_history