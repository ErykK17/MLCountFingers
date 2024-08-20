import cv2
import numpy as np
from keras import models

model = models.load_model('modelthresholded.keras')

def predict_fingers(img):
    img = cv2.resize(img, (128, 128))
    img = np.stack((img,) * 3, axis=-1) 
    img = np.expand_dims(img, axis=0) / 255.0 
    prediction = model.predict(img)
    return np.argmax(prediction) 


def process_frame(frame):
    print("test")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY) 

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        hand = threshold[y:y+h, x:x+w] 
        fingers_count = predict_fingers(hand) 
        cv2.putText(frame, f'Palce: {fingers_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    processed_frame = process_frame(frame)
    

    cv2.imshow("Finger Count", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
