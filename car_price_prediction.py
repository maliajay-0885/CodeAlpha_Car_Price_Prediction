
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("car data.csv")
df.columns = df.columns.str.strip()

target = "Selling_Price"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. Available columns: {df.columns.tolist()}"
    )

df = df.drop_duplicates()
df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = pd.to_numeric(df[target], errors="coerce")

valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

if len(df) < 10:
    raise ValueError("The dataset has too few records to train a reliable model.")

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()

for column in categorical_features:
    X[column] = X[column].astype("string").fillna("Unknown").astype(str)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numerical_features),
    ("categorical", categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nCAR PRICE PREDICTION MODEL")
print("=" * 35)
print("Dataset records:", len(X))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Algorithm: Random Forest Regressor")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.4f}")

comparison = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions
})
comparison.to_csv("outputs/prediction_results.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7)
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], "r--")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

importance_df.to_csv("outputs/feature_importance.csv", index=False)

plt.figure(figsize=(10, 7))
plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 15 Features Affecting Car Price Predictions")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

joblib.dump(model, "outputs/car_price_model.joblib")

print("\nTop influential features:")
print(importance_df.to_string(index=False))
print("\nFiles saved in the outputs folder.")
print("Model saved as outputs/car_price_model.joblib")
print("\nProject completed successfully.")
