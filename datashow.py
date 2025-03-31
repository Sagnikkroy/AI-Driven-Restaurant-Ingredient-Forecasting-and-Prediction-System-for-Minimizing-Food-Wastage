import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px
import pickle
import sys

# Load predicted data from predict.py
with open("predicted_data.pkl", "rb") as f:
    data = pickle.load(f)

ingredient_demand_df = data["ingredient_demand_df"]
monthly_sales = data["monthly_sales"]
month_to_predict = data["month_to_predict"]

# Visualize ingredient demand
fig1 = px.bar(ingredient_demand_df, x="Ingredient", y="Estimated Demand", title=f"Ingredient Demand for {month_to_predict}", labels={"Estimated Demand": "Quantity Required"})
fig1.show()

# Visualize monthly sales trend
fig2 = px.line(monthly_sales, x="Month", y="Quantity_Ordered", title="Monthly Sales Trend", labels={"Quantity_Ordered": "Total Orders"})
fig2.show()
