import cv2
import pytesseract
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the extracted text dataset
df_extracted = pd.read_csv('C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/extracted_texts.csv')

# Handle missing values by dropping rows with NaN in the 'text' column
df_extracted = df_extracted.dropna(subset=['text'])

# Step 1: Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_extracted['text'])

# Step 2: Set the target variable
y = df_extracted['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with class weights to handle imbalance
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Classification report with zero_division set to avoid warnings
print(classification_report(y_test, y_pred, zero_division=0))

# Load the test dataset
# test_df = pd.read_csv("C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/test1.csv")

# Transform the test data using the same vectorizer
X_test = vectorizer.transform(df_extracted['text'])

# Make predictions
predictions = model.predict(X_test)

# Assuming sample_test_out.csv gives you the format
sample_out = pd.read_csv("C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/sample_test_out.csv")

# Create a new DataFrame with the predictions
predictions_df = pd.DataFrame(predictions, columns=['prediction'])

# Make sure the length of the predictions matches the length of the index in the sample_out DataFrame
if len(predictions) < len(sample_out):
    # Add empty rows to the predictions_df to match the length of the sample_out DataFrame
    predictions_df = pd.concat([predictions_df, pd.DataFrame({'prediction': [None]*(len(sample_out) - len(predictions_df))})], ignore_index=True)
elif len(predictions) > len(sample_out):
    # Remove excess rows from the predictions_df to match the length of the sample_out DataFrame
    predictions_df = predictions_df.head(len(sample_out))

# Assign the predictions to the entity_value column
sample_out['prediction'] = predictions_df['prediction']

# Save the formatted file as test_out.csv
sample_out.to_csv("test_out.csv", index=False)