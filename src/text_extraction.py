import cv2
import pytesseract
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
# If you are on Windows, provide the path to the tesseract executable
# Update this path with your local installation of tesseract
pytesseract.pytesseract.tesseract_cmd = r"E:/program/tesseract.exe"

def extract_text_from_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to RGB format (if needed)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use Tesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(img_rgb)

    return extracted_text


def clean_text(text):
    # Remove unwanted characters and numbers that are not relevant
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep only alphanumeric and spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
    cleaned_text = cleaned_text.strip().lower()  # Convert to lowercase
    return cleaned_text

def convert_text_to_features(text_data):
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text data into numerical features
    text_features = vectorizer.fit_transform([text_data])

    return text_features, vectorizer

# Test the function
image_path = "C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/downloaded_image.jpg"  # Change this if needed
extracted_text = extract_text_from_image(image_path)

cleaned_text = clean_text(extracted_text)
print("Cleaned Text:")
print(cleaned_text)

text_features, vectorizer = convert_text_to_features(cleaned_text)
print("Numerical Features (TF-IDF):")
print(text_features.toarray())



# Load the dataset
train_df = pd.read_csv("C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/train1.csv")

# Extract features and labels
X = train_df['entity_name']  # Feature (Text Data)
y = train_df['entity_value']  # Labels (Target Variable)

# Clean and convert the text data to numerical features using the TF-IDF vectorizer
X_cleaned = [clean_text(text) for text in X]
X_features = vectorizer.transform(X_cleaned)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1}")
