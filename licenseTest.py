import re

import numpy as np
from PIL import Image
from ultralytics import YOLO
from pytesseract import pytesseract
import cv2


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def process_image(img):
    model = YOLO('C:/Users/meloo/Desktop/DL Webcam/runs/detect/train/weights/best.pt')

    img = np.array(Image.open(img))
    gray_image = grayscale(img)
    thresh, im_bw = cv2.threshold(gray_image, 90, 190, cv2.THRESH_BINARY)
    test_image = cv2.cvtColor(im_bw, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(np.uint8(test_image)).convert('RGB')
    PIL_image = Image.fromarray(test_image.astype('uint8'), 'RGB')

    results = model(PIL_image)
    predictions = results[0].boxes.data.tolist()
    print(predictions)
    label_names = ['Address', 'Class', 'DOB', 'Exp date', 'First name', 'Issue date', 'Last name', 'License number',
                   'Sex']
    PIL_image.show()
    text_dict = {}

    for prediction in predictions:
        x1, y1, x2, y2, confidence, label_no = prediction
        label_no = int(label_no)
        coordinates = [x1, y1, x2, y2, ]
        extracted_text = perform_ocr(PIL_image, coordinates)

        if len(extracted_text) != 0:
            label = label_names[label_no]

            cleaned_extracted_text = extracted_text.strip()
            cleaned_extracted_text = cleaned_extracted_text.replace('\n',' ')
            cleaned_extracted_text = re.sub(r'(?<=\d)\s+(?=\d)', '', cleaned_extracted_text)

            text_dict[label] = cleaned_extracted_text
    print(text_dict)
    return text_dict


def perform_ocr(image, coordinates, expansion=7):
    x1, y1, x2, y2 = map(int, coordinates)
    image_np = np.array(image)
    expanded_x1 = max(0, x1 - expansion)
    expanded_y1 = max(0, y1 - expansion)
    expanded_x2 = min(image_np.shape[1], x2 + expansion)
    expanded_y2 = min(image_np.shape[0], y2 + expansion)

    cropped_img = image_np[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
    cropped_img_pil = Image.fromarray(cropped_img)

    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract

    text = pytesseract.image_to_string(cropped_img_pil)
    return text
