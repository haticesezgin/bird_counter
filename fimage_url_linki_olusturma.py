from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# Resimlerin bulunduğu klasör
IMAGE_FOLDER = "images"


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
