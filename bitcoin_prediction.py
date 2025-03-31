# Analysis Algorithm: Bitcoin Prediction

#Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
occ_data = pd.read_csv('OccChainletsInTime.txt', delimiter='\t')
price_data = pd.read_csv('pricedBitcoin2009-2018.csv')

# Rename columns if needed to make sure they match
occ_data.rename(columns={'Year': 'year', 'Day': 'day'}, inplace=True)
price_data.rename(columns={'Year': 'year', 'Day': 'day'}, inplace=True)

# Merge the datasets on 'year' and 'day' to align chainlet data with Bitcoin prices
df = pd.merge(occ_data, price_data[['year', 'day', 'price']], on=['year', 'day'], how='inner')

# Feature engineering: Creating lagged price features
df['price_lag1'] = df['price'].shift(1)
df['price_lag2'] = df['price'].shift(2)
df['price_lag3'] = df['price'].shift(3)

# Drop any rows with missing values (created by shifting the price columns)
df.dropna(inplace=True)

# Optimize Correlation Visualization: Focus on a subset of features
features_to_plot = ['price', 'price_lag1', 'price_lag2', 'price_lag3'] + [col for col in df.columns if '1:' in col][:10]  # Top 10 '1:x' chainlet features
df_subset = df[features_to_plot]

# Visualizing Correlations between selected features
plt.figure(figsize=(12, 8))
correlation_matrix = df_subset.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Selected Features')
plt.show()

# Features (chainlet data and lagged price features) and target (price)
X = df.drop(columns=['price', 'year', 'day'])
y = df['price']

# Splitting the data: Use data up to November 30, 2017, for training and December 2017 for testing
train_data = df[df['year'] < 2017]
test_data = df[df['year'] == 2017]

# Separating features and target for training and testing
X_train = train_data.drop(columns=['price', 'year', 'day'])
y_train = train_data['price']
X_test = test_data.drop(columns=['price', 'year', 'day'])
y_test = test_data['price']

# Scaling features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression Model
ridge_model = Ridge(alpha=1.0)  # Regularization strength
ridge_model.fit(X_train_scaled, y_train)
ridge_predictions = ridge_model.predict(X_test_scaled)

# Neural Network Model (MLP) - Simplified with fewer hidden layers and max_iter increased
nn_model = MLPRegressor(hidden_layer_sizes=(5,), max_iter=500, random_state=42, warm_start=True)
nn_model.fit(X_train_scaled, y_train)
nn_predictions = nn_model.predict(X_test_scaled)

# Track NN loss during training
train_loss = nn_model.loss_curve_

# Evaluation: Calculate RMSE for both models
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))

print(f'Ridge Regression RMSE: {ridge_rmse}')
print(f'Neural Network RMSE: {nn_rmse}')

# Save predictions for December 2017 to CSV
predictions = pd.DataFrame({
    'date': test_data['year'].astype(str) + '-' + test_data['day'].astype(str),
    'predicted_price_ridge': ridge_predictions,
    'predicted_price_nn': nn_predictions
})

predictions.to_csv('predictions_december_2017.csv', index=False)

# Plot predictions to visualize model performance
plt.figure(figsize=(10, 6))
plt.plot(test_data['day'], y_test, label='Actual Prices', color='black')
plt.plot(test_data['day'], ridge_predictions, label='Ridge Predictions', color='blue')
plt.plot(test_data['day'], nn_predictions, label='NN Predictions', color='red')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Bitcoin Price Prediction: Ridge vs. Neural Network')
plt.legend()
plt.show()

# Additional Diagram: Feature Importance Visualization (for Ridge model coefficients)
# Display feature importance based on Ridge model coefficients
ridge_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(ridge_model.coef_)
})
ridge_importance = ridge_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(ridge_importance['Feature'][:10], ridge_importance['Importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Features by Importance in Ridge Regression')
plt.show()

# Neural Network Loss Curve Plot
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Neural Network Training Loss Curve')
plt.legend()
plt.show()






