# image video using open cv
import cv2

# read image
# processing
# closing

# image = cv2.imread('comet.jpg')
# # print(image)
# print(type(image))
# print(image.shape)
# cv2.imshow('Holland',image)
# cv2.waitKey()
# cv2.destroyAllWindows()

image = cv2.imread('comet.jpg')
grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('GrayScale', grayImage)
cv2.imshow('Holland',image)
print(grayImage)
print(grayImage.shape)
cv2.waitKey()
cv2.destroyAllWindows()