import cv2
import easyocr
import numpy as np
from numpy import ndarray


picture_read =easyocr.Reader(['en','ar'],gpu=True)

def ocr_scan(image_path:str) -> str:
    result = picture_read.readtext(str(image_path))
    recognized_text = " ".join(elem[1] for elem in result)

    return recognized_text

print(ocr_scan("sample2.jpeg"))

# for t in picture_results:
#   print (t[1])


import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO
from pytesseract import pytesseract
import cv2
import easyocr

picture_read =easyocr.Reader(['en','ar'],gpu=True)
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)

    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    image = cv2.medianBlur(image, 3)

    return image


def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image



# Load a model
def process_image(img):
    model = YOLO('C:/Users/meloo/Desktop/DL Webcam/runs/detect/train/weights/best.pt')  # pretrained YOLOv8n model

    img = np.array(Image.open(img))

    gray_image = grayscale(img)

    thresh, im_bw = cv2.threshold(gray_image, 90, 190, cv2.THRESH_BINARY)

    # Run batched inference on a list of images
    test_image = cv2.cvtColor(im_bw, cv2.COLOR_BGR2RGB)

    PIL_image = Image.fromarray(np.uint8(test_image)).convert('RGB')
    PIL_image = Image.fromarray(test_image.astype('uint8'), 'RGB')


    results = model(PIL_image)  # return a list of Results objects

    predictions = results[0].boxes.data.tolist()

    label_names = ['Address', 'Class', 'DOB', 'Exp date', 'First name', 'Issue date', 'Last name', 'License number',
                   'Sex']
    text_dict = {}
    for prediction in predictions:
        x1, y1, x2, y2, confidence, label_no = prediction
        label_no = int(label_no)
        coordinates = [x1, y1, x2, y2,]
        extracted_text = perform_ocr(PIL_image,coordinates)

        if len(extracted_text) != 0:
            label = label_names[label_no]
            text_dict[label] = extracted_text


# #################
#     predictions = results[0].boxes.data.tolist()
#     label_names = ['Address', 'Class', 'DOB', 'Exp date', 'First name', 'Issue date', 'Last name', 'License number', 'Sex']
#     drawn_image = Image.fromarray(test_image.copy())
#     draw = ImageDraw.Draw(drawn_image)
#     font = ImageFont.load_default()
#     for prediction in predictions:
#         x1, y1, x2, y2, confidence, label = prediction
#         label = int(label)
#
#         draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=5)
#
#         text = f"{label_names[label]} ({confidence:.3f})"
#         text_width, text_height = font.getsize(text)
#         text_x = x1 + 5
#         text_y = y1 + 5
#         draw.rectangle([(text_x, text_y), (text_x + text_width, text_y + text_height)], fill="red")
#         draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
#     drawn_image.show()
#     #########################
#     text_dict = {}
#
#     for r in results:
#         im_array = r.plot()  # plot a BGR numpy array of predictions
#         im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#
#         for i in range(len(r)):
#             extracted_text = perform_ocr(img, r.boxes.xyxy[i])
#
#             if len(extracted_text) != 0:
#                 label = r.names[i]
#                 text_dict[label] = extracted_text
#     print(text_dict)
    return text_dict


def perform_ocr(image, coordinates, expansion=7):
    x1, y1, x2, y2 = map(int, coordinates)

    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Expand the bounding box by a little
    expanded_x1 = max(0, x1 - expansion)
    expanded_y1 = max(0, y1 - expansion)
    expanded_x2 = min(image_np.shape[1], x2 + expansion)  # Use image_np.shape instead of image.shape
    expanded_y2 = min(image_np.shape[0], y2 + expansion)  # Use image_np.shape instead of image.shape

    # Crop the expanded area
    cropped_img = image_np[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

    # Convert numpy array to PIL Image
    cropped_img_pil = Image.fromarray(cropped_img)

    # path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # # Perform OCR using pytesseract
    # pytesseract.tesseract_cmd = path_to_tesseract
    # text = pytesseract.image_to_string(cropped_img_pil)

    result = picture_read.readtext(cropped_img)
    recognized_text = " ".join(elem[1] for elem in result)

    return recognized_text

def ocr_scan(image_path:str) -> str:
    result = picture_read.readtext(str(image_path))
    recognized_text = " ".join(elem[1] for elem in result)

    return recognized_text