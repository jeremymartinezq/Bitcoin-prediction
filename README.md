# Bitcoin Price Prediction

## Overview
This project implements a Bitcoin price prediction model using historical price data and chainlet-based features. Two machine learning models, Ridge Regression and a Neural Network (MLPRegressor), are utilized to predict Bitcoin prices. The dataset includes Bitcoin price data from 2009 to 2018.

## Features & Methods
- **Data Preprocessing:**
  - Loads and merges chainlet occurrence data (`OccChainletsInTime.txt`) and Bitcoin price data (`pricedBitcoin2009-2018.csv`).
  - Creates lagged price features (`price_lag1`, `price_lag2`, `price_lag3`) to incorporate historical trends.
  - Cleans and prepares data for modeling.

- **Feature Selection & Correlation Analysis:**
  - Selects relevant features and visualizes correlations using a heatmap.

- **Machine Learning Models:**
  - **Ridge Regression**: A linear model with L2 regularization to handle multicollinearity.
  - **Neural Network (MLPRegressor)**: A simple feedforward neural network for price prediction.
  - Data is split into training (pre-2017) and testing (December 2017) sets.
  - Features are scaled using `StandardScaler` to normalize input values.

- **Evaluation Metrics:**
  - RMSE (Root Mean Squared Error) is used to assess model performance.

- **Visualization:**
  - Price prediction comparison plot.
  - Ridge regression feature importance.
  - Neural network training loss curve.

## Installation & Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Running the Script
Run the script in a Python environment:
```bash
python bitcoin_prediction.py
```

## Output
- **RMSE Scores** for both models.
- **Predicted Bitcoin prices for December 2017** (saved as `predictions_december_2017.csv`).
- **Visualizations**:
  - Correlation matrix heatmap.
  - Price prediction comparison plot.
  - Feature importance bar chart.
  - Neural network loss curve.
  - ![image](https://github.com/user-attachments/assets/4b48ee33-c207-489d-8fde-bb49e0aae873)
  - ![image](https://github.com/user-attachments/assets/12cecd6b-225e-41b9-a610-48e9c0b30fe1)
  - ![image](https://github.com/user-attachments/assets/4da962b6-05ae-4a8a-afe2-785a8f81d238)
  - ![image](https://github.com/user-attachments/assets/2f5f9127-e3f6-491f-b62f-d0b1bd86eeba)
  - ![image](https://github.com/user-attachments/assets/324b5f7e-0b96-43ce-bdb6-3139bce8f58a)
  - ![image](https://github.com/user-attachments/assets/7926b0c5-c16a-44d9-b939-5919fa4d2d72)
  - ![image](https://github.com/user-attachments/assets/9c65d115-cc77-4438-b7ce-8e2773dd37f6)
  - ![image](https://github.com/user-attachments/assets/f36b55e4-d04b-4aa9-8a88-15db115863e9)

## Future Improvements
- Enhance the neural network architecture.
- Introduce additional time-series forecasting techniques.
- Optimize hyperparameters for better accuracy.

## Author
Developed by Jeremy Martinez-Quinones.

## License
This is licensed under MIT license - see LICENSE file for details.


