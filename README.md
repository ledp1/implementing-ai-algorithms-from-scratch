# Artificial Intelligence

This project contains Python implementations of linear regression and sales regression models, demonstrating fundamental machine learning concepts using NumPy and Matplotlib. All code is organized in the `machine-learning` folder for clarity and modularity.

## Project Structure

- `machine-learning/`
  - `linear_regression_analysis.py`: Implements linear regression from scratch on a simple dataset, with visualization.
  - `sales_regression_model_1.py`: Linear regression on advertising costs vs. sales, with visualization.
  - `sales_regression_model_2.py`: Linear regression on ad hours vs. weekly sales, with visualization.
  - `sales_regression_model_3.py`: (Empty, ready for future models.)
- `LICENSE`, `.gitignore`, `README.md`, `venv/`: Standard project files and virtual environment.

## Features

- Implements linear regression from first principles
- Uses NumPy for efficient numerical computations
- Visualizes data and regression results with Matplotlib
- Modular code for easy extension

## Requirements

- Python 3.x
- NumPy
- Matplotlib

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
   pip install numpy matplotlib
   ```

## Usage

Navigate to the `machine-learning` folder and run any script. For example:
```bash
cd machine-learning
python linear_regression_analysis.py
```

Each script will:
- Load sample data
- Calculate regression coefficients
- Print the resulting linear equation
- Display a plot of the data and regression line

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 