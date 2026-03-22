import cv2
import mediapipe as mp

# El modeli başlatılır
mp_hands = mp.solutions.hands
eller = mp_hands.Hands(max_num_hands=1)  # 1 el ile başlıyoruz
cizim = mp.solutions.drawing_utils

# Kamerayı aç
kamera = cv2.VideoCapture(0)

while True:
    ret, kare = kamera.read()
    if not ret:
        break

    # Görüntüyü BGR'den RGB'ye çevir
    kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)

    # MediaPipe ile el analizi
    sonuc = eller.process(kare_rgb)

    if sonuc.multi_hand_landmarks:
        for el in sonuc.multi_hand_landmarks:
            # Parmakları ekrana çiz
            cizim.draw_landmarks(kare, el, mp_hands.HAND_CONNECTIONS)

    # Görüntüyü göster
    cv2.imshow("El Tespiti", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()




