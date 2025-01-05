import os
from google.colab import drive

drive.mount("/content/drive")


###################### 101 #####################

from tensorflow.keras import datasets, layers, models
import numpy as np

mnist = datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)

print(X_train[0].shape)
#print(X_train[0])

from google.colab.patches import cv2_imshow

cv2_imshow(X_train[0])
print(y_train[0])

X_train_sc, X_test_sc = X_train / 255.0, X_test / 255.0
X_train_sc = X_train_sc.reshape((60000,28,28,1))
X_test_sc = X_test_sc.reshape((10000,28,28,1))

print(X_train.shape)
print(X_test.shape)

###################### 102 #####################

model1 = models.Sequential()
model1.add(layers.Flatten(input_shape=(28,28)))
model1.add(layers.Dense(512, activation='relu'))
model1.add(layers.Dropout(0.2))
model1.add(layers.Dense(10, activation='softmax'))

model1.summary()

model2 = models.Sequential()
model2.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))
model2.add(layers.MaxPooling2D((2,2)))
model2.add(layers.Conv2D(64,(3,3),activation='relu'))
model2.add(layers.MaxPooling2D((2,2)))
model2.add(layers.Conv2D(64,(3,3),activation='relu'))
model2.add(layers.Flatten())
model2.add(layers.Dense(64,activation='relu'))
model2.add(layers.Dense(10,activation='softmax'))

model2.summary()

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(X_train_sc, y_train, epochs=10)

################## 103 ###############
model1_test_loss, model1_test_acc = model1.evaluate(X_test_sc, y_test)
print(model1_test_acc)

################## 104 ###############
predictions = model1.predict(X_train_sc)

print(predictions.shape)

print(predictions[0])
print( np.argmax(predictions[0]) )
print(y_train[0])

################## 102 103 104 for model2 ###############

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(X_train_sc, y_train, epochs=10)

model2_test_loss, model2_test_acc = model2.evaluate(X_test_sc, y_test)
print(model2_test_acc)

predictions = model2.predict(X_train_sc)

predictions = model2.predict(X_train_sc)

print(predictions.shape)

print(predictions[0])
print( np.argmax(predictions[0]) )
print(y_train[0])

