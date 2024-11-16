import pandas as pd

# Path to your dataset
file_path = './data/dataset.csv'  # Ensure the file path matches your setup

# Load the dataset using pandas
df = pd.read_csv(file_path)

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display column names
print("\nColumn names:")
print(df.columns)

# Display dataset information
print("\nDataset Info:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())
