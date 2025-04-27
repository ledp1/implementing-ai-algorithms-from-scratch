import numpy as np
import matplotlib.pyplot as plt

# Step 1: Get the data set
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Step 2: Compute the mean of the X and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Step 3: Calculate the coefficients
m = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x)**2)
c = mean_y - m * mean_x

# Voila! We have our model
print(f"Model: y = {c} + {m}*x")  # Output: Model: y= 2.2 + 0.6*x

# Plot the data and the model
plt.scatter(x, y, color="red", marker="o", s=30)  # s is marker size
y_pred = c + m * x
plt.plot(x, y_pred, color="blue")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Analysis')
plt.show() # Run the code to see the plot