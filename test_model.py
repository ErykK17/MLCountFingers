import cv2
import numpy as np
from keras import models

model = models.load_model('model.keras')


def predict_fingers(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    print(prediction)
    return np.argmax(prediction)

# Przykład predykcji
image_path = 'C:/Users/erykr/OneDrive/Pulpit/MLFingerCount/MLCountFingers/TestImages/0.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (128, 128))
cv2.imshow("idk",img)
cv2.waitKey(3000)
cv2.destroyAllWindows()
print(f'Predicted number of fingers: {predict_fingers(image_path)}')