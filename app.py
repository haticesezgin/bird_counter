from flask import Flask, request, jsonify
import subprocess
import os
import time
import uuid
import requests

app = Flask(__name__)

# YOLOv5 modelinin yolu
YOLO_PATH = "yolov5"
WEIGHTS_PATH = "yolov5s.pt"


@app.route('/count-birds', methods=['POST'])
def count_birds_api():
    data = request.json  # JSON formatında veri alıyoruz
    if 'file_url' not in data or 'question' not in data:
        return jsonify({"error": "Lütfen bir dosya URL'si ('file_url') ve soru ('question') sağlayın."}), 400

    file_url = data['file_url']
    question = data['question']

    # Dosyayı indir
    unique_id = str(uuid.uuid4())[:8]
    file_path = f"temp_{unique_id}.jpeg"

    try:
        # Resmi URL'den indir
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            return jsonify({"error": "Resim indirilemedi."}), 400
    except Exception as e:
        return jsonify({"error": f"Resim indirilirken hata oluştu: {str(e)}"}), 500

    # YOLOv5 ile kuşları tespit et
    start_time = time.time()
    exp_folder = f"runs/detect/exp_{unique_id}"

    try:
        subprocess.run(
            [
                "python3",
                os.path.join(YOLO_PATH, "detect.py"),
                "--weights", WEIGHTS_PATH,
                "--source", file_path,
                "--conf", "0.25",
                "--save-txt",
                "--exist-ok",
                "--project", "runs/detect",
                "--name", f"exp_{unique_id}"
            ],
            capture_output=True,
            text=True
        )
    except Exception as e:
        return jsonify({"error": f"YOLOv5 çalıştırılırken hata oluştu: {str(e)}"}), 500

    end_time = time.time()
    processing_time = end_time - start_time

    # Tespit edilen kuş sayısını bul
    labels_folder = f"{exp_folder}/labels"
    bird_count = 0

    if os.path.exists(labels_folder):
        label_files = [f for f in os.listdir(
            labels_folder) if f.endswith(".txt")]
        for label_file in label_files:
            with open(os.path.join(labels_folder, label_file), "r") as f:
                bird_count += len(f.readlines())

    # Geçici dosyayı temizle
    os.remove(file_path)

    # Sonucu string olarak döndür
    result_string = f"Soru: {question}\nResimde {bird_count} kuş bulundu.\nİşlem süresi: {processing_time:.2f} saniye."

    return jsonify({"response": result_string})


if __name__ == '__main__':
    app.run(debug=True)
