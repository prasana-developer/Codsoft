# Import Libraries
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("advertising.csv")

# Display Dataset
print(df.head())
print(df.info())
print(df.describe())

# Check Missing Values
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV vs Sales")
plt.show()

# Select Feature and Target
X = df[["TV"]]
y = df["Sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Parameters
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)

# Regression Line
plt.figure(figsize=(6,4))
plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("Simple Linear Regression")
plt.show()

# Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Predict for New Data
new_tv = [[150]]
predicted_sales = model.predict(new_tv)

print(f"Predicted Sales for TV advertising = 150: {predicted_sales[0]:.2f}")