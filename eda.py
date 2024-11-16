import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed dataset
df = pd.read_csv('./data/processed_dataset.csv')

# Basic information about the data
print(df.info())
print(df.describe())

# Visualize feature distributions
df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Visualize correlations between features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
