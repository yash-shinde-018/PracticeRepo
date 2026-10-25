
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load dataset
data = pd.read_csv("sales.csv")
print(data.head())

# 2. Separate input and output
X = data[["TV"]]
y = data["Sales"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Data scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict sales
y_pred = model.predict(X_test)
print("Actual Sales:", y_test.values)
print("Predicted Sales:", y_pred)

# 7. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
