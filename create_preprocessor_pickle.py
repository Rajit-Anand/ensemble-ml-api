# create_preprocessor_pickle.py

import pandas as pd
import pickle
from preprocessor import DataPreprocessor

# Define the column lists (use your actual column names)
numerical_columns = [
    "Age", "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction", "Work/Study Hours",
    "Financial Stress"
]
one_hot_columns = [
    "Gender", "Working Professional or Student", "City", "Family History of Mental Illness"
]
label_columns = [
    "Degree", "Profession", "Dietary Habits", "Have you ever had suicidal thoughts ?", "Sleep Duration"
]

# Load your training data
# For example, load from a CSV file:
df = pd.read_csv("train.csv")  # Update the filename/path as needed

# Create an instance of DataPreprocessor and fit it on the training data
preprocessor = DataPreprocessor(numerical_columns, one_hot_columns, label_columns)
preprocessor.fit(df)

# Save (pickle) the fitted preprocessor
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("Preprocessor pickle file created successfully.")
