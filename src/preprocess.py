import cv2
import requests
import os
import pytesseract
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r"E:/program/tesseract.exe"

def download_image(image_url, save_path):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download image from {image_url}")
    
    
# Function to extract text from image
def extract_text_from_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to RGB format (if needed)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use Tesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(img_rgb)

    return extracted_text


def preprocess_image(image_path):
    """
    Preprocess the image to make it more OCR-friendly.
    Steps include: grayscale conversion, noise reduction, and thresholding.
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian Blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Convert the image to binary (black and white)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Return the preprocessed image
    return img

def save_preprocessed_image(image, output_path):
    """
    Save the preprocessed image to the output path.
    """
    cv2.imwrite(output_path, image)
    
    


if __name__ == "__main__":
    # Example: Download an image, preprocess it, and save it
    df = pd.read_csv('C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/train.csv')
    
    extracted_texts = []
    labels = []


   # Loop through the dataset, process each image, and extract text
for index, row in df.iterrows():
    image_url = row['image_link']  # Adjust column name as per your CSV
    label = row['entity_name']  # Adjust column name as per your CSV
    
    try:
        # Download the image
        downloaded_image_path = f"C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/downloaded_image_{index}.jpg"
        download_image(image_url, downloaded_image_path)
        
        # Extract text from the image
        extracted_text = extract_text_from_image(downloaded_image_path)
        
        # Store the extracted text and corresponding label
        extracted_texts.append(extracted_text)
        labels.append(label)
        
        # Optional: Remove image after processing to save disk space
        os.remove(downloaded_image_path)
        
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")

# Store extracted texts and labels for further processing
df_extracted = pd.DataFrame({
    'text': extracted_texts,
    'label': labels
})


# Save the extracted data for model training
df_extracted.to_csv('C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/extracted_texts.csv', index=False)
print("All images processed and texts extracted successfully.")
