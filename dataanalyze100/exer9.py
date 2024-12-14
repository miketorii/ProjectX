import os
from google.colab import drive

drive.mount("/content/drive")


import cv2
from google.colab.patches import cv2_imshow

#im = cv2.imread("/content/drive/MyDrive/Lena.jpg")
#print(type(im))
#print(im.shape)

img = cv2.imread("/content/drive/MyDrive/chap9/img/img01.jpg")
print(img.shape)
cv2_imshow(img)
