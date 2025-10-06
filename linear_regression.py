# 1. Check and import required libraries
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    print("All libraries imported successfully")

except ImportError as e:
    print("Missing library:", e.name)

# 2. Load the dataset (from CSV source equivalent to Kaggle version)
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# 3. Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# 4. Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# 5. Visualize relationships
sns.pairplot(df, x_vars=df.columns.drop("medv"), y_vars="medv", height=2.5)
plt.suptitle("Pairwise plots with target (medv)", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 6. Split features and target
X = df.drop("medv", axis=1)  # features
y = df["medv"]                # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:")
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat}: {coef}")

# 8. Predictions & evaluation
y_pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

# 9. Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=3)
plt.xlabel("Actual medv")
plt.ylabel("Predicted medv")
plt.title("Actual vs Predicted")
plt.show()

# 10. Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, edgecolors=(0, 0, 0))
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel("Predicted medv")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

