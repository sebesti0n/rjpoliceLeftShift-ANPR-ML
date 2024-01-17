from ultralytics import YOLO
from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
def index():
    image_filename = 'image1.jpeg'
    image_url = url_for('static', filename=image_filename)
    yolo = YOLO('best.pt')
    detection = yolo.predict('image5.jpeg',save=True)
    print(detection)

    return render_template('index.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
