import re  # Import the regular expression module for text processing
import numpy as np  # Import NumPy for numerical operations
from PIL import Image  # Import Image module from Python Imaging Library (PIL) for image processing
from ultralytics import YOLO  # Import YOLO object detection model from ultralytics library
from pytesseract import pytesseract  # Import pytesseract for OCR (Optical Character Recognition)
import cv2  # Import OpenCV library for image processing


def grayscale(image):
    """
    Convert an image to grayscale.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: Grayscale version of the input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def process_image(img):
    """
    Process the input image to detect objects using YOLO model and perform OCR.

    Args:
        img (str): Path to the input image file.

    Returns:
        dict: Dictionary containing extracted text with corresponding labels.
    """
    # Initialize YOLO model
    model = YOLO('C:/Users/meloo/Desktop/DL Webcam/runs/detect/train/weights/best.pt')

    # Read input image
    img = np.array(Image.open(img))

    # Convert image to grayscale
    gray_image = grayscale(img)

    # Convert grayscale image to RGB format
    test_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

    # Convert numpy array to PIL Image
    PIL_image = Image.fromarray(test_image).convert('RGB')

    # Perform object detection
    results = model(PIL_image)
    predictions = results[0].boxes.data.tolist()

    # Define label names for the detected objects
    label_names = ['Address', 'Class', 'DOB', 'Exp date', 'First name', 'Issue date', 'Last name', 'License number',
                   'Sex']

    # Initialize dictionary to store extracted text with corresponding labels
    text_dict = {}

    # Iterate through each prediction
    for prediction in predictions:
        x1, y1, x2, y2, confidence, label_no = prediction
        label_no = int(label_no)
        coordinates = [x1, y1, x2, y2]

        # Perform OCR on the cropped region of interest
        extracted_text = perform_ocr(PIL_image, coordinates)

        # If text is extracted, associate it with the corresponding label
        if len(extracted_text) != 0:
            label = label_names[label_no]

            # Clean the extracted text
            cleaned_extracted_text = extracted_text.strip()
            cleaned_extracted_text = cleaned_extracted_text.replace('\n', ' ')
            cleaned_extracted_text = re.sub(r'(?<=\d)\s+(?=\d)', '', cleaned_extracted_text)

            # Store cleaned text with corresponding label in the dictionary
            text_dict[label] = cleaned_extracted_text

    # Print the extracted text and return the dictionary
    print(text_dict)
    return text_dict


def perform_ocr(image, coordinates, expansion=7):
    """
    Perform OCR on the cropped region of interest.

    Args:
        image (PIL.Image.Image): Input image in PIL format.
        coordinates (list): List containing coordinates of the region of interest [x1, y1, x2, y2].
        expansion (int): Expansion factor for cropping region of interest.

    Returns:
        str: Extracted text from the region of interest.
    """
    x1, y1, x2, y2 = map(int, coordinates)

    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Expand the region of interest
    expanded_x1 = max(0, x1 - expansion)
    expanded_y1 = max(0, y1 - expansion)
    expanded_x2 = min(image_np.shape[1], x2 + expansion)
    expanded_y2 = min(image_np.shape[0], y2 + expansion)

    # Crop the region of interest
    cropped_img = image_np[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

    # Convert cropped numpy array to PIL Image
    cropped_img_pil = Image.fromarray(cropped_img)

    # Set path to Tesseract executable
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract

    # Perform OCR on the cropped image
    text = pytesseract.image_to_string(cropped_img_pil)

    return text
