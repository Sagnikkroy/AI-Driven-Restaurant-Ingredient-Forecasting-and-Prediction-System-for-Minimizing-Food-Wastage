import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import subprocess

import pickle

# Run mergeddata.py to load data into memory
mergeddata = {}
exec(open("mergeddata.py").read(), mergeddata)

# Retrieve in-memory DataFrames
final_df = mergeddata["final_df"]
final_wastage_df = mergeddata["final_wastage_df"]

# Normalize month names for consistency
final_df["Month"] = final_df["Month"].str.strip().str.capitalize()
final_wastage_df["Month"] = final_wastage_df["Month"].str.strip().str.capitalize()

# Ask user for the month to predict
month_to_predict = input("Enter the month to predict (e.g., January, February): ").strip().capitalize()

# Filter past data for the selected month
past_orders = final_df[final_df["Month"] == month_to_predict]
past_wastage = final_wastage_df[final_wastage_df["Month"] == month_to_predict]

if past_orders.empty or past_wastage.empty:
    print("No exact historical data available for this month. Using overall yearly trends instead.")
    past_orders = final_df
    past_wastage = final_wastage_df

# Prepare data for trend analysis
monthly_sales = final_df.groupby("Month")["Quantity_Ordered"].sum().reset_index()
monthly_sales["Month_Index"] = np.arange(len(monthly_sales))

# Train Linear Regression model to identify sales trend
X = monthly_sales[["Month_Index"]]
y = monthly_sales["Quantity_Ordered"]
reg_model = LinearRegression()
reg_model.fit(X, y)
predicted_growth = reg_model.predict([[len(monthly_sales)]])[0]  # Predict next month sales trend

# Ingredient-wise demand calculation
ingredient_demand = {}
for _, row in past_orders.iterrows():
    dish = row["Dish_Ordered"]
    quantity = row["Quantity_Ordered"]
    ingredients = row["Ingredients"].split(", ") if pd.notna(row["Ingredients"]) else []
    for ingredient in ingredients:
        ingredient_demand[ingredient] = ingredient_demand.get(ingredient, 0) + quantity

# Convert to DataFrame
ingredient_demand_df = pd.DataFrame(list(ingredient_demand.items()), columns=["Ingredient", "Estimated Demand"])

# Train ML model for confidence score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
confidence_score = r2_score(y_test, y_pred)

# Print Predictions
print("\nInventory Prediction for", month_to_predict)
print(ingredient_demand_df)
print(f"Predicted sales trend: {predicted_growth:.2f} units")
print(f"Confidence Score of prediction: {confidence_score:.2f}")




# Save predicted data for visualization
predicted_data = {
    "ingredient_demand_df": ingredient_demand_df,
    "monthly_sales": monthly_sales,
    "month_to_predict": month_to_predict
}

with open("predicted_data.pkl", "wb") as f:
    pickle.dump(predicted_data, f)

# Call datashow.py to visualize the predictions
subprocess.run(["python", "datashow.py"])
