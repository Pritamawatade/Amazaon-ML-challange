import cv2
import requests
import os

def download_image(image_url, save_path):
    """
    Download image from a URL and save it to a specified path.
    """
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise ValueError(f"Failed to download image from {image_url}")
    return save_path

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

    # URL from the test.csv file (replace this with an actual URL from your test.csv)
    image_url = "https://m.media-amazon.com/images/I/417NJrPEk+L.jpg"
    
    # Local file path to save the downloaded image
    downloaded_image_path = "C:/Users/PRITAM/Downloads/66e31d6ee96cd_student_resource_3/student_resource 3/dataset/downloaded_image.jpg"

    # Download the image from the URL
    download_image(image_url, downloaded_image_path)

    # Preprocess the downloaded image
    preprocessed_img = preprocess_image(downloaded_image_path)

    # Save the preprocessed image
    output_image_path = "../dataset/preprocessed_image.jpg"
    save_preprocessed_image(preprocessed_img, output_image_path)

    print(f"Preprocessed image saved at: {output_image_path}")
