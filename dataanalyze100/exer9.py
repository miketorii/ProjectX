import os
from google.colab import drive

drive.mount("/content/drive")

###### 81 ######

import cv2
from google.colab.patches import cv2_imshow

#im = cv2.imread("/content/drive/MyDrive/Lena.jpg")
#print(type(im))
#print(im.shape)

img = cv2.imread("/content/drive/MyDrive/chap9/img/img01.jpg")
print(img.shape)
cv2_imshow(img)

#### 82
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

cap = cv2.VideoCapture("/content/drive/MyDrive/chap9/mov/mov01.avi")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(width)
print(height)
print(count)
print(fps)

num = 0
num_frame = 100
list_frame = []
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    list_frame.append(frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    if num > num_frame:
      break
  num = num+1

print("------done-------")
cap.release()

plt.figure()
patch = plt.imshow(list_frame[0])
plt.axis('off')
def animate(i):
  patch.set_data(list_frame[i])
anim = FuncAnimation(plt.gcf(), animate, frames=len(list_frame), interval=1000/30.0)
plt.close()

HTML(anim.to_jshtml())
