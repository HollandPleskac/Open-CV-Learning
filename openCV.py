import cv2
import matplotlib.pyplot as plt

# Gray Scale

# img = cv2.imread('comet.jpg')
# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Comet',img)
# cv2.imshow('Gray Scale Comet',gray_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Resize

# img = cv2.imread('comet.jpg')
# resized_img = cv2.resize(img, (300,200))
# cv2.imshow('Comet',img)
# cv2.imshow('Resized Comet',resized_img)
# print('Original Image Pixels',img.shape)
# print('Resized Image Pixels',resized_img.shape)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Eroding (reducing object features) 

# img = cv2.imread('eating-food.jpg')
# eroded_img = cv2.erode(img, (5,5))
# cv2.imshow('Image',img)
# cv2.imshow('Eroded Image',eroded_img)
# print('Original Image Pixels',img)
# print('Eroded Image Pixels',eroded_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Dilating - increases the object area and is used to account for missing pixel data while reading
# 

# img = cv2.imread('comet.jpg')
# eroded_img = cv2.dilate(img, (5,5))
# cv2.imshow('Image',img)
# cv2.imshow('Dilated Image',eroded_img)
# print('Original Image Pixels',img)
# print('Eroded Image Pixels',eroded_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# img = cv2.imread('letters.png')
# img = cv2.resize(img, (200,200))
# dilated_img = cv2.dilate(img, (5,5))
# plt.imshow(img)
# plt.show()
# cv2.imshow('Image',img)
# cv2.imshow('Dilated Image',dilated_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Rectangles around contoured image of numbers

# img = cv2.imread('numbers.jpg')
# img = cv2.resize(img, (600,600))
# cv2.rectangle(img, (300,300), (400,400), (150,23,123), 2)
# cv2.putText(img, 'Some Text', (300,300),cv2.FONT_HERSHEY_PLAIN,4,(232,34,212),3)

# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# val, threshold_img = cv2.threshold(gray_img, 90, 255, cv2.THRESH_BINARY_INV)
# contours,var = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours,-1,(0,255,255), 3)
# print(contours)

# rectangles = []
# for c in contours:
#   numberData = cv2.boundingRect(c)
#   print(numberData[0])
#   cv2.rectangle(img, (numberData[0]-5,numberData[1]-5), (numberData[0]+numberData[2]+5,numberData[1]+numberData[3]+5), (0,255,0), 2)
  # +- 5 is an offset to make the squares bigger
  # draw a rectange around each image
  # make the rectangle a little bit bigger

# output of for c in coutours print c
# (80, 217, 43, 56)
# x1, y1, width, height
# cv2.rectangle wants x1, y1, x2, and y2
# to get x2 add width to x1, to get y2 add height to y1

# print(img)
# print(val)
# cv2.imshow('Numbers', gray_img)
# cv2.imshow('contours', img)
# cv2.imshow('threshold image', threshold_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# OpenCV Video Project
# each frame of the video can be treated like an image ( every operation/function for imgs can be used on a video frame )

# cap = cv2.VideoCapture(0) # can be an existing video file or port number of webcam
# while True: # show video forever
#   ret,frame = cap.read() # read the video feed frame by frame
#   if ret:
#     cv2.imshow("Video", frame) # show video feed
#     gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Gray Video", gray_frame)
#     if cv2.waitKey(1)==ord('q'): # if q key is pressed
#       break
# cap.release() # stop capture
# cv2.destroyAllWindows() # close all windows


# Rectangle 800x600 total
# color frame[y1:y2,x1:x2] = (0,0,255)

# cap = cv2.VideoCapture(0) # can be an existing video file or port number of webcam
# cap.set(3,800) # x,y,width,height width is 3rd
# cap.set(4,600) # height is 4th
# while True: # show video forever
#   ret,frame = cap.read() # read the video feed frame by frame
#   if ret:
#     roi = frame[0:600, 0:400]
#     flipped = cv2.flip(roi,0)
#     frame[0:600,0:400] = flipped
#     cv2.rectangle(frame, (0,0), (400,600), (150,23,123), 2)
#     cv2.imshow("Video", frame) # show video feed
#     # cv2.imshow("Half", roi)
#     # cv2.imshow("Flipped", flipped)
#     if cv2.waitKey(1)==ord('q'): # if q key is pressed
#       break
# cap.release() # stop capture
# cv2.destroyAllWindows() # close all windows

# --------------------------------------------------------------------- FEATURE DETECTION --------------------------------------------------------------

# Face Detection Image

# img = cv2.imread('2 people.jfif')
# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# faces = faceCascade.detectMultiScale(gray_img, 1.3, 5) # min zoom: 1.3x, max zoom: 5x
# for face in faces:
#   print(face)
#   x1 = face[0]
#   y1 = face[1]
#   width = face[2]
#   height = face[3]
#   cv2.rectangle(img, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
# print(faces)
# cv2.imshow('People',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Eye Detection Image

# img = cv2.imread('myself.PNG')
# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faceCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# eyes = faceCascade.detectMultiScale(gray_img, 1.3, 5) # min zoom: 1.3x, max zoom: 5x
# for eye in eyes:
#   print(eye)
#   x1 = eye[0]
#   y1 = eye[1]
#   width = eye[2]
#   height = eye[3]
#   cv2.rectangle(img, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
# print(eyes)
# cv2.imshow('Holland',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Smile Detection Image

# img = cv2.imread('2 people.jfif')
# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faceCascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# smiles = faceCascade.detectMultiScale(gray_img, 1.3, 5) # min zoom: 1.3x, max zoom: 5x
# for smile in smiles:
#   print(smile)
#   x1 = smile[0]
#   y1 = smile[1]
#   width = smile[2]
#   height = smile[3]
#   cv2.rectangle(img, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
# print(smile)
# cv2.imshow('People',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Face Detection Video

# cap = cv2.VideoCapture(0) # can be an existing video file or port number of webcam
# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# while True: # show video forever
#   ret,frame = cap.read() # read the video feed frame by frame
#   if ret:
#     gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray_frame, 1.3, 5) # min zoom: 1.3x, max zoom: 5x
#     for face in faces:
#       print(face)
#       x1 = face[0]
#       y1 = face[1]
#       width = face[2]
#       height = face[3]
#       cv2.rectangle(frame, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
#       cv2.rectangle(gray_frame, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
#     cv2.imshow("Video", frame) # show video feed
#     cv2.imshow("Gray Video", gray_frame)
#     if cv2.waitKey(1)==ord('q'): # if q key is pressed
#       break
# cap.release() # stop capture
# cv2.destroyAllWindows() # close all windows

# Eye Detection Video

# cap = cv2.VideoCapture(0) # can be an existing video file or port number of webcam
# faceCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# while True: # show video forever
#   ret,frame = cap.read() # read the video feed frame by frame
#   if ret:
#     gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     eyes = faceCascade.detectMultiScale(gray_frame, 1.3, 5) # min zoom: 1.3x, max zoom: 5x
#     for eye in eyes:
#       # print(eye)
#       x1 = eye[0]
#       y1 = eye[1]
#       width = eye[2]
#       height = eye[3]
#       cv2.rectangle(frame, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
#       cv2.rectangle(gray_frame, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
#     cv2.imshow("Video", frame) # show video feed
#     cv2.imshow("Gray Video", gray_frame)
#     if cv2.waitKey(1)==ord('q'): # if q key is pressed
#       break
# cap.release() # stop capture
# cv2.destroyAllWindows() # close all windows

# Smile Detection Video

# cap = cv2.VideoCapture(0) # can be an existing video file or port number of webcam
# faceCascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# while True: # show video forever
#   ret,frame = cap.read() # read the video feed frame by frame
#   if ret:
#     gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     smiles = faceCascade.detectMultiScale(gray_frame, 1.3, 5) # min zoom: 1.3x, max zoom: 5x
#     for smile in smiles:
#       # print(smile)
#       x1 = smile[0]
#       y1 = smile[1]
#       width = smile[2]
#       height = smile[3]
#       cv2.rectangle(frame, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
#       cv2.rectangle(gray_frame, (x1,y1), (x1+width,y1+height), (150,23,123), 2)
#     cv2.imshow("Video", frame) # show video feed
#     cv2.imshow("Gray Video", gray_frame)
#     if cv2.waitKey(1)==ord('q'): # if q key is pressed
#       break
# cap.release() # stop capture
# cv2.destroyAllWindows() # close all windows

# Smile Detection with Face Detection Video

cap = cv2.VideoCapture(0) # can be an existing video file or port number of webcam
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True: # show video forever
  ret,frame = cap.read() # read the video feed frame by frame
  if ret:
    gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_frame, 1.3, 5) # min zoom: 1.3x, max zoom: 5x
    for face in faces:
      x1 = face[0]
      y1 = face[1]
      width = face[2]
      height = face[3]
      roi = gray_frame[y1:y1+height,x1:x1+width]
      # print(x1, y1, width, height)
      cv2.rectangle(frame, (x1,y1), (x1+width,y1+height), (120,129,123), 2)
      smiles = smileCascade.detectMultiScale(roi, 1.3, 5) # min zoom: 1.3x, max zoom: 5x
      for smile in smiles:
        x1_smile = smile[0] + x1
        y1_smile = smile[1] + y1
        width_smile = smile[2]
        height_smile = smile[3]
        print(x1_smile, y1_smile, width_smile, height_smile)
        cv2.rectangle(frame, (x1_smile,y1_smile), (x1_smile+width_smile,y1_smile+height_smile), (0,229,123), 2)
      cv2.imshow('ROI', roi)
    cv2.imshow("Video", frame) # show video feed
      
    if cv2.waitKey(1)==ord('q'): # if q key is pressed
      break
cap.release() # stop capture
cv2.destroyAllWindows() # close all windows

