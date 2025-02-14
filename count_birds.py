import cv2
import time
import numpy as np  # NumPy modülünü içe aktar


def count_birds(image):
    start_time = time.time()

    # Önceden eğitilmiş modeli yükle
    prototxt = "deploy.prototxt"
    model = "mobilenet_iter_73000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Sınıf etiketleri
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    # Resmi model için hazırla
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Modeli çalıştır
    net.setInput(blob)
    detections = net.forward()

    # Kuşları say ve görsel üzerinde işaretle
    bird_count = 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.1:  # Güven eşiğini 0.1 olarak ayarladık
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] == "bird":
                bird_count += 1

                # Tespit edilen kuşun sınırlayıcı kutusunu al
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Kuşu görsel üzerinde işaretle
                cv2.rectangle(image, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)
                label = f"Bird {bird_count}: {confidence * 100:.2f}%"
                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end_time = time.time()
    processing_time = end_time - start_time

    # İşaretlenmiş görseli kaydet
    cv2.imwrite("output.jpg", image)

    return bird_count, processing_time, "Resim başarıyla işlendi."
