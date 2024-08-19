import cv2
import os
import numpy as np

from keras import models, layers
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    labels = []
    images = []
    for label in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, label)):
            img_path = os.path.join(data_dir, label, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

images, labels = load_data('/home/eryk/Pulpit/ML_IloscPalcow/Fingers/')
images = images / 255.0


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')  # Wyjścia: 0, 1, 2, 3, 4, 5
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

