import cv2
import numpy as np
from keras import models

# Ładowanie modelu
model = models.load_model('model.keras')

# Funkcja do przewidywania liczby palców
def predict_fingers(img):
    img = cv2.resize(img, (128, 128))  # Dopasowanie rozmiaru do modelu
    img = np.stack((img,) * 3, axis=-1)  # Konwersja do 3 kanałów
    img = np.expand_dims(img, axis=0) / 255.0  # Normalizacja
    prediction = model.predict(img)
    return np.argmax(prediction)  # Zwraca etykietę z największym prawdopodobieństwem

# Funkcja do wykrywania dłoni i liczenia palców
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konwersja do odcieni szarości
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Wygładzanie obrazu
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)  # Binaryzacja

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Znalezienie największego konturu
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        hand = threshold[y:y+h, x:x+w]  # Wycięcie obszaru z dłonią
        fingers_count = predict_fingers(hand)  # Przewidywanie liczby palców
        cv2.putText(frame, f'Palce: {fingers_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rysowanie prostokąta wokół dłoni
    
    return frame

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Przetwarzanie klatki
    processed_frame = process_frame(frame)
    
    # Wyświetlenie wyniku
    cv2.imshow("Finger Count", processed_frame)
    
    # Wyjście po wciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
