import numpy as np

X = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

y = np.array([56, 81, 119, 22, 103], dtype='float32')

# Add a column of ones to the X matrix to represent the intercept term
ones = np.ones(shape=(len(X), 1))
X = np.append(ones, X, axis=1)