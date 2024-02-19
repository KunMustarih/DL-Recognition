from flask import Flask, render_template,request
from licenseTest import perform_ocr,process_image
app = Flask(__name__, static_url_path='/static')


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/process_image', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['image']
        License_data = process_image(f)
        return render_template('result.html', processed_values=License_data)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
