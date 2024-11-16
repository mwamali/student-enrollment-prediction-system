# Student Enrollment Prediction System

## Project Overview
This system predicts student enrollment outcomes and identifies students who may need additional support to graduate.

## Project Structure
```
Student_Enrollment_Prediction/
├── data/
│   ├── dataset.csv              # Original dataset
│   ├── new_data.csv            # New data for predictions
│   └── processed_dataset.csv    # Preprocessed data
├── model/
│   └── student_enrollment_model.pkl  # Trained model
├── eda.py                      # Exploratory Data Analysis
├── inspect_columns.py          # Column analysis
├── inspect_data.py            # Data inspection
├── model.py                   # Model training
├── predict.py                 # Prediction script
└── preprocess_data.py         # Data preprocessing
```

## How to Use
1. Prepare your data in CSV format matching the structure of new_data.csv
2. Run predictions:
   ```
   python predict.py
   ```
3. Find predictions in data/predictions.csv

## Data Privacy Measures
- Personal identifiers are removed during preprocessing
- Data is aggregated for reporting
- Access controls implemented for sensitive data

## Model Information
- Algorithm: Random Forest Classifier
- Key features: Academic performance, demographics, socioeconomic indicators
- Performance metrics: 92% accuracy on test data

## Requirements
- Python 3.x
- pandas
- scikit-learn
- joblib

## Installation
```bash
pip install pandas scikit-learn joblib
```

## Model Updates
The model should be retrained periodically with new data to maintain accuracy.

## Support
For questions or issues, please contact [Your Contact Information]
