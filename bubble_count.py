import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit
from PIL import Image

X = 10
open_size = 30
baw_thread = 80

#Reading image in black and white
img = Image.open("100XOptison6.jpg")
width, height = img.size
print(width,height)
gray = img.convert('L')
plt.imshow(gray, cmap='gray')
plt.show()
bw = gray.point(lambda x: 255 if x<baw_thread else 0, '1')
plt.imshow(bw, cmap='gray')
plt.show()
img = np.asarray( bw, dtype="uint8" )
original = img

#Resize
#img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    
#Filling holes

contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    cv2.drawContours(img,[cnt],0,255,-1)
fill_hole = img

#Opening
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_size,open_size))
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(open_size/4),int(open_size/4)))
subtract = cv2.morphologyEx(img - opening, cv2.MORPH_OPEN, kernel2)

plt.subplot(221)
plt.imshow(original, cmap='gray')
plt.subplot(222)
plt.imshow(opening, cmap='gray')
plt.subplot(223)
plt.imshow(subtract, cmap='gray')
plt.show()

img = opening
img = cv2.blur(img, (3,3))
subtract = cv2.blur(subtract, (3,3))

#Hough
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,35, param1=1,param2=21,minRadius=0,maxRadius=200)

#Big ones
circles = np.uint16(np.around(circles))
res = np.zeros(5000)
count=0
for i in circles[0,:]:
    res[count] = i[2]*0.07/660/5*X
    count+=1
    cv2.circle(img,(i[0],i[1]),10,(0,0,255),3)
print(count)
plt.imshow(img)
plt.show()

#Small ones
circles_small = cv2.HoughCircles(subtract,cv2.HOUGH_GRADIENT,1,10, param1=1,param2=13,minRadius=0,maxRadius=200)
circles_small = np.uint16(np.around(circles_small))
for i in circles_small[0,:]:
    res[count] = i[2]*0.07/660/5*X
    count+=1
    cv2.circle(subtract,(i[0],i[1]),5,(0,0,255),2)        

print(count)
plt.imshow(subtract)
plt.show()

#plt.hist(res[res!=0], bins = 30)
#plt.title("Distribution")
#plt.xlabel("Size (mm)")
#plt.ylabel("Frequency")
#plt.show()

#Display
plt.subplot(221)
plt.imshow(original, cmap='gray')
plt.subplot(222)
plt.imshow(img, cmap='gray')
plt.subplot(223)
plt.imshow(subtract, cmap='gray')
plt.subplot(224)
plt.hist(res[res!=0], bins = 30)
plt.title("Distribution")
plt.xlabel("Size (mm)")
plt.ylabel("Frequency")

plt.show()
