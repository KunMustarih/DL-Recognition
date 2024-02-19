## Project Structure

- **index.html**: Main HTML file containing the upload form for ID images.
- **result.html**: HTML template for displaying extracted ID details.
- **styles.css**: CSS file for styling the user interface.
- **licenseTest.py**: Python script containing image processing, object detection, and OCR functions.
- **app.py**: Flask application script for handling web requests.
- **README.md**: Documentation outlining project details, architecture, and technologies used.

## Technologies Used

- **Flask**: Python web framework for building the backend server.
- **HTML/CSS**: Frontend markup and styling for the web application.
- **JavaScript**: Client-side scripting for dynamic interactions.
- **OpenCV**: Library for image processing tasks such as grayscale conversion and cropping.
- **PyTesseract**: Python wrapper for Google's Tesseract-OCR Engine for text extraction.
- **YOLO (You Only Look Once)**: Object detection model for identifying regions of interest in images.

## Assumptions and Considerations

- The application assumes that users will provide clear and well-oriented images of their driver's licenses.
- It is assumed that the YOLO model is trained to recognize relevant regions of interest (e.g., name, address) in the ID images accurately.

## Obstacles and Considerations for Future Improvement

### Limited Dataset

The project faced a challenge due to the limited dataset available for training and testing. With only 600 images, the model's accuracy might be limited. Obtaining a larger and more diverse dataset could significantly improve performance.

### Image Editing and Processing

Each driver's license may require different levels of editing and processing for optimal extraction. Tasks such as normalization, skew correction, scaling, noise removal, thinning, skeletonization, grayscale conversion, thresholding, or binarization are essential but challenging to standardize due to the variations in license designs. Continued research and experimentation are needed to determine the optimal editing levels for different types of licenses, ultimately enhancing OCR accuracy.

### OCR Accuracy

While the object detection model may accurately identify regions of interest, the OCR component's ability to translate images to text may introduce errors. Improvements in OCR algorithms or post-processing techniques could mitigate these issues and enhance overall accuracy.

### Installation and Usage
To install and run the Driver's License Scanner Web Application based on the provided code, follow the instructions below:

### Prerequisites
Python 3.6 or higher installed on your system.
pip package manager.
Installation Steps
Clone the repository to your local machine using Git:

Install the required Python packages that are provided in the requirements.txt file

Ensure you have Tesseract-OCR installed on your system. You can download it from the official repository [link](https://github.com/UB-Mannheim/tesseract/wiki).

Change the path in this variable path_to_tesseract that is in line 114 in licenseTest.py to the path where the tesseract.exe is installed on your PC.

Place the YOLOv8 model weights file (e.g., best.pt found in runs/detect/train/weights/best.pt) in a directory accessible by the code. Update the path in licenseTest.py accordingly if needed. In my code its model = YOLO('C:/Users/meloo/Desktop/DL Webcam/runs/detect/train/weights/best.pt') at line 33. Hence change it to where it is saved in your PC.

Running the Application
Start the Flask server by running the following command --> python app.py
Open your web browser and go to http://localhost:8000.

You will see a page with an upload form. Upload an image of your driver's license using the form.

Wait for the processing to complete. Once done, the extracted details from the driver's license will be displayed on the screen.

To stop the Flask server, press Ctrl + C in the terminal where it is running.

Additional Notes
Ensure that your webcam is properly connected and accessible by the browser for capturing images.
Depending on your operating system and environment setup, you may need to install additional dependencies for OpenCV and Tesseract-OCR. Please refer to the respective documentation for installation instructions.
For optimal performance, use images of driver's licenses with good lighting and clear text.
