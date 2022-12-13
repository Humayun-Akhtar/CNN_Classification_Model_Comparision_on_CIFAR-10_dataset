import numpy as np
import cv2 

width, height = 32,32
img = np.zeros((height,width)) 
location = 0
shade = 0
# for i in range(50):

cv2.imwrite("my_image.png", img)
cv2.imshow("my_image.png", img)