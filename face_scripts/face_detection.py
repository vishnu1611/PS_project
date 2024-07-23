import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

image_path='/home/user/VScode_PS1/outdoor-portrait-of-beautiful-young-woman-european-city-landscape-on-background-lausanne-switzerland-2R5KAKC.jpg'

#converting to numpy

img=cv2.imread(image_path)

print(img.shape)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray_image.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


#scale factor 1.1 looks for face 10 percent lesser than before
#min neighbors low value = false +ves, high value= missed +ves
#minsize is minsize super lame
face=face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))


for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

#back to colour
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')



