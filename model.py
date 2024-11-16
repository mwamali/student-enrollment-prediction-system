import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the processed dataset
df = pd.read_csv('./data/processed_dataset.csv')

# Define the target variable and features
X = df.drop('Target_Graduate', axis=1)  # Drop 'Target_Graduate' from features
y = df['Target_Graduate']  # Use 'Target_Graduate' as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model for later use
import joblib
joblib.dump(model, './model/student_enrollment_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


