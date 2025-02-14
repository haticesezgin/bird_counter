from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


def count_birds_in_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bird_count = len(contours)
    return bird_count


@app.route('/count_birds', methods=['POST'])
def count_birds():
    file = request.files['file']
    file_path = './uploaded_image.jpg'
    file.save(file_path)

    # Kuşları sayma
    bird_count = count_birds_in_image(file_path)

    return jsonify({'answer': f'Resimde {bird_count} kuş var.'})


if __name__ == '__main__':
    app.run(debug=True)
