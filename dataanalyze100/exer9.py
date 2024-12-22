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

######### 83 ##############
cap = cv2.VideoCapture("/content/drive/MyDrive/chap9/mov/mov01.avi")
num = 0
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret:
    filepath = "/content/drive/MyDrive/chap9/snapshot/snapshot_" + str(num) + ".jpg"
    cv2.imwrite(filepath, frame)
  num = num + 1
  if num >= count:
    break

cap.release()
cv2.destroyAllWindows()

#################### 84
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hogParams = { 'winStride': (8,8), 'padding': (32,32), 'scale' : 1.05, 'hitThreshold' : 0, 'groupThreshold' : 5 }

img = cv2.imread("/content/drive/MyDrive/chap9/img/img01.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
human, r = hog.detectMultiScale( gray, **hogParams )
if( len(human)>0 ):
  for (x,y,w,h) in human:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 3)

cv2_imshow(img)
cv2.imwrite("/content/drive/MyDrive/chap9/temp.jpg",img)

###################### 85
cascade_file = "/content/drive/MyDrive/chap9/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

img = cv2.imread("/content/drive/MyDrive/chap9/img/img02.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_list = cascade.detectMultiScale(gray, minSize=(50,50))

for (x, y, w, h) in face_list:
  color = (0, 0, 255)
  pen_w = 3
  cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=pen_w)

cv2_imshow(img)
cv2.imwrite("/content/drive/MyDrive/chap9/temp2.jpg",img)

############## 86
import dlib
import math

predictor = dlib.shape_predictor("/content/drive/MyDrive/chap9/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

img = cv2.imread("/content/drive/MyDrive/chap9/img/img02.jpg")
dets = detector(img, 1)

for k, d in enumerate(dets):
  shape = predictor(img, d)

  color_f = (0,0,255)
  color_l_out = (255,0,0)
  color_l_in = (0,255,0)
  line_w = 3
  circle_r = 3
  fontType = cv2.FONT_HERSHEY_SIMPLEX
  fontSize = 1
  cv2.rectangle(img, (d.left(), d.top()), (d.right(),d.bottom()), color_f, line_w)
  cv2.putText(img, str(k), (d.left(), d.top()), fontType, fontSize, color_f, line_w)

  num_of_points_out = 17
  num_of_points_in = shape.num_parts - num_of_points_out
  gx_out = 0
  gy_out = 0
  gx_in = 0
  gy_in = 0

  for shape_point_count in range(shape.num_parts):
    shape_point = shape.part(shape_point_count)

    if shape_point_count < num_of_points_out:
      cv2.circle(img, (shape_point.x, shape_point.y), circle_r, color_l_out, line_w)
      gx_out = gx_out + shape_point.x/num_of_points_out
      gy_out = gy_out + shape_point.y/num_of_points_out
    else:
      cv2.circle(img, (shape_point.x, shape_point.y), circle_r, color_l_in, line_w)
      gx_in = gx_in + shape_point.x/num_of_points_in
      gy_in = gy_in + shape_point.y/num_of_points_in
  
  cv2.circle(img, (int(gx_out), int(gy_out)), circle_r, (0,0,255), line_w)
  cv2.circle(img, (int(gx_in), int(gy_in)), circle_r, (0,0,0), line_w)

  theta = math.asin(2*(gx_in-gx_out)/(d.right()-d.left()))
  radian = theta*180/math.pi
  print(theta)
  print(radian)

  if radian<0:
    textPrefix = " left "
  else:
    textPrefix = " right "
  
  textShow = textPrefix + str( round(abs(radian), 1)) + " deg."
  cv2.putText(img, textShow, (d.left(), d.top()), fontType, fontSize, color_f, line_w)

cv2_imshow(img)
cv2.imwrite("/content/drive/MyDrive/chap9/temp3.jpg", img)

######################87

cap = cv2.VideoCapture("/content/drive/MyDrive/chap9/mov/mov01.avi")
width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hogParams = { 'winStride': (8,8), 'padding': (32,32), 'scale': 1.05, 'hitThreshold': 0, 'groupThreshold': 5}

#movie_name = "timelapse.avi"
movie_name = "/content/drive/MyDrive/chap9/timelapse.avi"
fourcc =cv2.VideoWriter_fourcc('X','V','I','D')
video = cv2.VideoWriter(movie_name, fourcc, 30, (width, height) )

num = 0

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret:
    if(num%10==0):
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      human, r = hog.detectMultiScale(gray, **hogParams)
      if(len(human)>0):
        for (x,y,w,h) in human:
          cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 3)

      video.write(frame)

  else:
    break

  num = num + 1

video.release()
cap.release()
cv2.destroyAllWindows()

print("-------------done--------------")

################################### 88

import pandas as pd

cap = cv2.VideoCapture("/content/drive/MyDrive/chap9/mov/mov01.avi")
fps = cap.get(cv2.CAP_PROP_FPS)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hogParams = { 'winStride': (8,8), 'padding': (32,32), 'scale': 1.05, 'hitThreshold': 0, 'groupThreshold': 5}

num = 0
list_df = pd.DataFrame( columns=['time','people'] )
while(cap.isOpened()):
  ret, frame = cap.read()
  #print(ret)
  if ret:
    if (num%10==0):
      #print("--------calc-----------")
      gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
      human, r = hog.detectMultiScale(gray, **hogParams)
      if(len(human)>0):
        for (x,y,w,h) in human:
          cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 3)

      tmp_se = pd.Series( [num/fps, len(human)], index=list_df.columns )
      #list_df = list_df.append( tmp_se, ignore_index=True )
      list_df = pd.concat([list_df, pd.DataFrame([tmp_se])], ignore_index=True)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  else:
    break
      
  num = num+1
  #print(num)

cap.release()
cv2.destroyAllWindows()
print("---------------done----------------")

print(list_df.head())

import matplotlib.pyplot as plt

plt.plot(list_df['time'], list_df['people'])
plt.xlabel('time(sec.)')
plt.ylabel('population')
plt.ylim(0,15)
plt.show()


