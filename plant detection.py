import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('Leafspotdisease.jpg')
if img is None:
    print("Error: Image not found.")
    exit()

# Convert BGR to RGB for accurate color representation
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert RGB to HSV for color segmentation
hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

# Define HSV range for healthy green color
lower_green = np.array([25, 0, 20])
upper_green = np.array([100, 255, 255])
green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

# Define HSV range for diseased brown color
lower_brown = np.array([10, 0, 10])
upper_brown = np.array([30, 255, 255])
brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)

# Combine masks to get the final mask
final_mask = cv2.bitwise_or(green_mask, brown_mask)

# Apply the final mask to the original image
final_result = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)

# Calculate the percentage of diseased area
total_pixels = final_mask.size
diseased_pixels = cv2.countNonZero(brown_mask)
disease_percentage = (diseased_pixels / total_pixels) * 100

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("HSV Image")
plt.imshow(hsv_img)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Green Mask (Healthy Areas)")
plt.imshow(green_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Brown Mask (Diseased Areas)")
plt.imshow(brown_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Final Mask")
plt.imshow(final_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Final Result")
plt.imshow(final_result)
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Percentage of Diseased Area: {disease_percentage:.2f}%")
