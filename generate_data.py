import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

num_rows = 500

data = {
    'Transaction_ID': range(1001, 1001 + num_rows),
    'Date': [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(num_rows)],
    'Customer_Age': np.random.normal(35, 12, num_rows).astype(int),
    'Gender': np.random.choice(['Male', 'Female', 'Non-Binary', np.nan], num_rows, p=[0.45, 0.45, 0.05, 0.05]),
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Toys'], num_rows),
    'Purchase_Amount': np.random.exponential(150, num_rows).round(2)
}

df = pd.DataFrame(data)

# Introduce some messy data for our AI to clean!
# 1. Add missing values to Customer_Age
df.loc[np.random.choice(df.index, 30, replace=False), 'Customer_Age'] = np.nan

# 2. Add extreme outliers to Purchase_Amount (simulating bulk corporate buys or errors)
df.loc[np.random.choice(df.index, 5, replace=False), 'Purchase_Amount'] = [5000.50, 7500.00, 8200.75, 9500.00, 12000.99]

# 3. Add duplicate rows (simulating double-clicked purchases)
duplicates = df.sample(10)
df = pd.concat([df, duplicates], ignore_index=True)

# Overwrite test_data.csv
df.to_csv('data/test_data.csv', index=False)
print(f"data/test_data.csv generated successfully with {len(df)} rows!")
