import cv2
import numpy as np

# Modified to work in headless environment
img = np.zeros((100, 100, 3), dtype=np.uint8)
print("Created test image with shape:", img.shape)

# Save instead of display (headless-friendly)
cv2.imwrite('/tmp/test_image.png', img)
print("OpenCV test completed successfully - image saved!")

# Verify the image was saved
import os
if os.path.exists('/tmp/test_image.png'):
    print("✅ Image file created successfully")
else:
    print("❌ Failed to create image file")