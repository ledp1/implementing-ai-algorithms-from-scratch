import numpy as np
import matplotlib.pyplot as plt

# Step 1: Get the data set
advertising_costs = np.array([100, 200, 300, 400, 500])
sales = np.array([300, 500, 600, 700, 800])

# Step 2: Compute the mean of the X and y
mean_adv = np.mean(advertising_costs)
mean_sales = np.mean(sales)

# Step 3: Calculate the coefficients
m = np.sum((advertising_costs - mean_adv) * (sales - mean_sales)) / np.sum((advertising_costs - mean_adv)**2)
c = mean_sales - m * mean_adv

# Step 4: Print the model
print(f"Model: Sales = {c:.2f} + {m:.2f}*Advertising_Costs")

# Step 5: Plot the data and the model
plt.scatter(advertising_costs, sales, color = 'blue')
plt.plot(advertising_costs, c + m * advertising_costs, color = 'red')
plt.xlabel('Advertising Costs')
plt.ylabel('Sales')
plt.title('Advertising Costs vs Sales')
plt.show() # Run the code to see the plot