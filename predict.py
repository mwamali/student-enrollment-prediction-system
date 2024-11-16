import pandas as pd
import joblib


def preprocess_new_data(df):
    """
    Preprocess the new data to match the format expected by the model
    """
    # Fill missing values with appropriate defaults
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Fill numeric columns with median
    for col in numeric_columns:
        df[col] = df[col].fillna(0)

    # Fill categorical columns with mode
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    return df


def main():
    # Load the trained model
    try:
        model = joblib.load('./model/student_enrollment_model.pkl')
    except FileNotFoundError:
        print("Error: Model file not found. Please ensure the model file exists in the correct location.")
        return

    # Load and preprocess new data
    try:
        new_data = pd.read_csv('./data/new_data.csv')
        print("Original data shape:", new_data.shape)

        # Preprocess the data
        processed_data = preprocess_new_data(new_data)
        print("Processed data shape:", processed_data.shape)

        # Make predictions
        predictions = model.predict(processed_data)

        # Add predictions to the dataframe
        new_data['Predictions'] = predictions

        # Save predictions
        output_path = './data/predictions.csv'
        new_data.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

        # Display first few predictions
        print("\nFirst few predictions:")
        print(new_data[['Marital status', 'Course', 'Age at enrollment', 'Predictions']].head())

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()