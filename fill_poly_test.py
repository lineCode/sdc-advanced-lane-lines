import numpy as np
import matplotlib.pyplot as plt
import cv2

img = np.zeros((10, 10))
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x1 =    [3, 3, 3, 4, 4, 5, 5, 4, 3, 2]
x2 =    [7, 7, 7, 7, 8, 8, 9, 8, 8, 9]

print("vstack")
pts1 = np.vstack([x1, y])
pts2 = np.vstack([x2, y])
print(pts1)
print(pts2)

print("transposes")
pts1 = np.transpose(pts1)
pts2 = np.transpose(pts2)
print(pts1)
print(pts2)

print("flip 2")
pts2 = np.flipud(pts2)

print("array")
pts1 = np.array([pts1])
pts2 = np.array([pts2])
print(pts1)
print(pts2)

print("hstack")
pts = np.hstack((pts1, pts2))
print(pts)

print("int")
pts = np.int_([pts])
print(pts)

print("fill poly")
cv2.fillPoly(img, pts, 255)

plt.imshow(img)
plt.show()
