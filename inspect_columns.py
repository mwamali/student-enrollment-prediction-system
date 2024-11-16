import pandas as pd

# Load the processed dataset
df = pd.read_csv('./data/processed_dataset.csv')

# Print column names
print("Column Names in the Dataset:")
print(df.columns)
