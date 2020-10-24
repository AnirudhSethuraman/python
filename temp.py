import numpy as np
import cv2 
import matplotlib.pyplot as plt
#%matplotlib inline
#  Loading the image to be tested
test_image = cv2.imread('anirudh.jpg')

# Converting to grayscale as opencv expects detector takes in input gray scale images
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Displaying grayscale image
plt.imshow(test_image_gray, cmap='gray')
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))
for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
     #convert image to RGB and show image
plt.imshow(convertToRGB(test_image))