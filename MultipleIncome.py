
# Multiple Linear Regression using income.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("income.csv")

print("First five rows:")
print(df.head())

# i. Data scaling
X = df[["age", "experience"]]
y = df["income"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# iii. Training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ii. Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict income
y_pred = model.predict(X_test)

# iv. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Display actual and predicted income
result = pd.DataFrame({
    "Actual Income": y_test.values,
    "Predicted Income": y_pred
})

print("\nActual vs Predicted Income:")
print(result.head())

# Graph: Actual vs Predicted income
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="blue", label="Predicted Income")

# Reference line for perfect predictions
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot([min_value, max_value],
         [min_value, max_value], color="red",
         label="Perfect Prediction")

plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.title("Multiple Linear Regression - Income Prediction")
plt.legend()
plt.grid(True)
plt.show()
