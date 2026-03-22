import cv2

# Kamerayı başlat (0: varsayılan kamera)
kamera = cv2.VideoCapture(0)

# Kameradan sürekli görüntü al
while True:
    ret, kare = kamera.read()  # 'ret' başarıyı gösterir, 'kare' görüntüdür
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    cv2.imshow("Kamera Görüntüsü", kare)  # Görüntüyü göster

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereyi kapat
kamera.release()
cv2.destroyAllWindows()
