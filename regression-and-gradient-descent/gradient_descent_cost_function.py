import numpy as np

# The cost function in gradient descent. It is the calculation of the mean square error. 
def cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/m) * np.sum(np.square(predictions-y)) # Compute mean square error
    return cost