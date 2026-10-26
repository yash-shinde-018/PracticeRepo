
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("sales.csv")

print(df.head())

# i. Data scaling
X = df[["fahrenheit"]]
y = df["sales"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# iii. Training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ii. Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# iv. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Graph 1: Regression line
plt.figure(figsize=(8, 5))
plt.scatter(X["fahrenheit"], y, color="blue", label="Actual Data")

x_line = np.linspace(
    X["fahrenheit"].min(), X["fahrenheit"].max(), 100
).reshape(-1, 1)

x_line_scaled = scaler.transform(x_line)
y_line = model.predict(x_line_scaled)

plt.plot(x_line, y_line, color="red", label="Regression Line")
plt.xlabel("Fahrenheit")
plt.ylabel("Sales")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Graph 2: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="green")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()
