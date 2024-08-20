import cv2
import numpy as np
from keras import models


image = cv2.imread('TestImages/5T.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_, threshold = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)

model = models.load_model('modelthresholded.keras')


def predict_fingers(img):
    img = cv2.resize(img, (128, 128))
    img = np.stack((img,) * 3, axis=-1)  # Convert to 3 channels
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    print(prediction)
    return np.argmax(prediction)

img = cv2.resize(threshold, (128, 128))
cv2.imshow("idk",img)
cv2.waitKey(3000)
cv2.destroyAllWindows()
print(f'Predicted number of fingers: {predict_fingers(threshold)}')
