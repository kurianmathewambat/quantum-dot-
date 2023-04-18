
import cv2 as cv
import numpy as np
W = 600
import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread('testimage.tiff', cv2.IMREAD_GRAYSCALE)

# Threshold the image to create a binary image
_, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to clean up the binary image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw circles around the contours
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(img, center, radius, (255, 0, 0), 2)

# Display the image with circles around the dots
cv2.imshow('Detected Dots', img)
cv2.moveWindow('Detected Dots', 0, 1)
cv2.waitKey(0)

# Calculate the average intensity of two dots
# Assuming the two dots are the first and second dots detected
dot1_mask = np.zeros_like(img)
dot2_mask = np.zeros_like(img)
cv2.drawContours(dot1_mask, contours, 0, 255, -1)
cv2.drawContours(dot2_mask, contours, 1, 255, -1)
dot1_intensity = cv2.mean(img, mask=dot1_mask)[0]
dot2_intensity = cv2.mean(img, mask=dot2_mask)[0]
print("Average intensity of Dot 1:", dot1_intensity)
print("Average intensity of Dot 2:", dot2_intensity)

cv2.destroyAllWindows()