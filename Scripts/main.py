import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset
file_path = '../data/ConsumoCo2.csv'
dataset = pd.read_csv(file_path)

# Select features and target
features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']
target = 'CO2EMISSIONS'

X = dataset[features]
y = dataset[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict using the Linear Regression model
y_pred_linear = linear_model.predict(X_test)

# Train the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict using the KNN model
y_pred_knn = knn_model.predict(X_test)

# Evaluate the models
linear_metrics = {
    'R2 Score': r2_score(y_test, y_pred_linear),
    'Mean Squared Error': mean_squared_error(y_test, y_pred_linear),
    'Mean Absolute Error': mean_absolute_error(y_test, y_pred_linear)
}

knn_metrics = {
    'R2 Score': r2_score(y_test, y_pred_knn),
    'Mean Squared Error': mean_squared_error(y_test, y_pred_knn),
    'Mean Absolute Error': mean_absolute_error(y_test, y_pred_knn)
}

# Print the results
print("Linear Regression Metrics:")
print(linear_metrics)
print("\nKNN Metrics:")
print(knn_metrics)
