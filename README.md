# Artificial Intelligence

A Python implementation of linear regression from scratch using NumPy. This project demonstrates fundamental machine learning concepts by implementing a simple linear regression model without relying on machine learning libraries.

## Features

- Implements linear regression from first principles
- Uses NumPy for efficient numerical computations
- Demonstrates the mathematical foundation of linear regression
- Clean, well-documented code

## Requirements

- Python 3.x
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/luisdepombo/AI-pytorch.git
cd AI-pytorch
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy
```

## Usage

Run the script:
```bash
python linear_regression_analysis.py
```

The script will:
1. Load sample data
2. Calculate the regression coefficients
3. Print the resulting linear equation

## How It Works

The implementation follows these steps:
1. Calculate the mean of X and y values
2. Compute the slope (m) using the covariance formula
3. Calculate the y-intercept (c)
4. Form the linear equation: y = c + m*x

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 