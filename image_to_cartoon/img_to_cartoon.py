import cv2
import numpy as np

image = cv2.imread('picture.jpg')
image_blurr = cv2.stackBlur(image, (3, 3), 3)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

inverted_image = cv2.bitwise_not(gray_image)

blur_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)

inverted_blur = cv2.bitwise_not(blur_image)

sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)

kernel = np.ones((7, 7), np.uint8)
dilated_sketch = cv2.dilate(sketch, kernel, iterations=1)

sketch_color = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

mask = np.zeros_like(image)
mask[dilated_sketch == 255] = [0, 0, 0]

combined_img = cv2.addWeighted(image, 0.7, sketch_color, 0.3, 0)
combined_img = cv2.add(combined_img, mask)

cv2.imshow("gray_image", gray_image)
cv2.imshow("inverted_blur", inverted_blur)
cv2.imshow("sketch.png", sketch)
cv2.imshow("cartoon.png", combined_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
