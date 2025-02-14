import cv2
import numpy as np


def count_birds_in_image(image_path):
    # Görseli yükle
    img = cv2.imread(image_path)

    # Görseli gri tonlamaya dönüştür
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Görseli bulanıklaştır (gürültü azaltma)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Görselde kenarları tespit et
    edges = cv2.Canny(blurred, 50, 150)

    # Kontur tespiti yap
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kuş sayısını ver
    bird_count = len(contours)

    # Görseli göster
    # Konturları yeşil renkte çiz
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # Sonuçları yazdır
    print(f"Resimdeki kuş sayısı: {bird_count}")

    # Görseli ekranda göster (görseli kaydetmek de isteyebilirsiniz)
    cv2.imshow('Birds Detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bird_count
