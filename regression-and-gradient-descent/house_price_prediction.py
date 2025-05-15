import numpy as np

# multiple linear regression analysis
# Coefficients for a pretend real estate market model
# b0: intercept, b1: area, b2: age, b3: # bathrooms
# ntercept means the price of the house when the area, age, and # bathrooms are 0.
# Assume b0, b1 for area, b2 for age, b3 for # bathrooms.
# Shape of coefficients must be (4,) because there are 4 coefficients.
# Coefficients are the weights of the features.
coefficients = np.array([50000, 3000, -2000, 15000])  

# Data for a new house: [intercept, area, age, # bathrooms]
new_house = np.array([1, 150, 10, 2])

# Calculate the predicted price by dot multiplying coefficients and house features, which gives a scalar.
# Shape of coefficients: (4,). Shape of new_house: (4,). Shape of predicted_price: ().
predicted_price = np.dot(coefficients, new_house) 

# Print the predicted price
# Format the price with commas and two decimal places
print(f"Predicted house price: ${predicted_price:,.2f}") # Predicted house price: $510,000.00

