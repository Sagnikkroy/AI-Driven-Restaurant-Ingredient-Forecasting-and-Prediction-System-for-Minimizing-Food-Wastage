import pandas as pd

# Load datasets
orders_df = pd.read_csv(r"D:\PBL\messy_orders_customer_data.csv")
ingredients_df = pd.read_csv(r"D:\PBL\Dish_Name_Ingredients.csv")
holidays_df = pd.read_csv(r"D:\PBL\Holiday_List.csv")
wastage_df = pd.read_csv(r"D:\PBL\food_wastage2022_dataset.csv")

# Strip column names to remove extra spaces
holidays_df.columns = holidays_df.columns.str.strip()
wastage_df.columns = wastage_df.columns.str.strip()

# Convert Date columns to datetime format for proper merging
holidays_df["Date"] = pd.to_datetime(holidays_df["Date"], errors='coerce').dt.date
wastage_df["Date"] = pd.to_datetime(wastage_df["Date"], errors='coerce').dt.date

# Merge orders with ingredients on Dish_Ordered
merged_df = orders_df.merge(ingredients_df, left_on="Dish_Ordered", right_on="Dish_Name", how="left")

# Select relevant columns
final_df = merged_df[["Customer_ID", "Month", "Dish_Ordered", "Quantity_Ordered", "Order_Time", "Ingredients"]]

# Process holiday and wastage data
holidays_df["Holiday"] = 1  # Mark all holiday dates as 1
merged_wastage_df = wastage_df.merge(holidays_df[['Date', 'Holiday']], on="Date", how="left")
merged_wastage_df["Holiday"].fillna(0, inplace=True)  # Fill non-holiday dates with 0

# Select relevant columns
final_wastage_df = merged_wastage_df[["Date", "Month", "Wastage (kg)", "Holiday"]]

print("Data is stored in memory and ready for ML usage.")






print("First few rows of final_df:")
print(final_df.head())  # View first few rows of merged customer-orders data

print("\nFirst few rows of final_wastage_df:")
print(final_wastage_df.head())  # View first few rows of merged wastage data