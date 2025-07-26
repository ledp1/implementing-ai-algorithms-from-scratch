import numpy as np

# Generate data based on the form y=ax+b with some noise
X = 2 * np.random.rand(100,1)
y = 4 +3 * X+np.random.randn(100,1)

lr = 0.01 # Learning Rate
n_iter = 1000 # Max number of iterations
theta = np.random.randn(2,1) # Randomly initialized parameters
X_b = np.c_[np.ones((len(X),1)),X] # add bias parameter to X
theta, cost_history, theta_history = gradient_descent(X_b,y,theta,lr,n_iter)