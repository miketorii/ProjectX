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
