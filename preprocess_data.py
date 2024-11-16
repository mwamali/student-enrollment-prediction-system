import pandas as pd


def preprocess_data(input_file_path):
    """
    Preprocess the dataset: handle missing values, encode categorical variables, and save processed data.

    Args:
        input_file_path (str): Path to the raw dataset (CSV).

    Returns:
        str: Path to the processed dataset.
    """
    # Load the dataset
    df = pd.read_csv(input_file_path)

    # Handle missing values (fill numeric columns with their mean)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Save the preprocessed data
    processed_path = './data/processed_dataset.csv'
    df_encoded.to_csv(processed_path, index=False)

    print(f"Preprocessed data saved to {processed_path}")
    return processed_path


# Run the preprocessing function when the script is executed
if __name__ == '__main__':
    input_path = './data/dataset.csv'  # Path to your raw dataset
    preprocess_data(input_path)
