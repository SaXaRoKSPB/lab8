import cv2

image = cv2.imread("variant-7.jpg")

flip_image = cv2.flip(image, 1)

hight, length = image.shape[:2]
center = (hight // 2, length // 2)

matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated_image = cv2.warpAffine(flip_image, matrix, (hight, length))
cv2.imshow("Rotated image", rotated_image)
cv2.waitKey(0)
