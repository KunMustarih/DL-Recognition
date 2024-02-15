from flask import Blueprint, render_template

from distutils.log import debug
from fileinput import filename
from flask import *

app = Flask(__name__)
views = Blueprint(__name__,"views")

@views.route("/")
def home():
    return render_template("index.html")

@views.route('/process_image', methods=['POST'])
def process_image():
    # Handle image processing here using your Python script
    # Dummy example for demonstration:

    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
    processed_values = {'Name': 'John Doe', 'DOB': '01/01/1990', 'ID Number': '123456789'}
    return render_template('result.html', processed_values=processed_values)