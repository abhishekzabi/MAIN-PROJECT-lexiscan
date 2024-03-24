import cv2
import numpy as np

# Variables
drawing = False  # True if mouse is pressed
mode = True  # If True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1

# Function to handle mouse events
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, mode, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (255, 255, 255), 40)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (255, 255, 255), 40)

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Set the window name and callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:  # Press 'Esc' to exit
        break

# Shrink the image to 29 by 29 with INTER_AREA interpolation
shrunken_img = cv2.resize(img, (29, 29), interpolation=cv2.INTER_AREA)

# Print the size of the shrunken image
height, width, channels = shrunken_img.shape
print(f"Shrunken Image Size: Width = {width}, Height = {height}, Channels = {channels}")

# Save the shrunken image
cv2.imwrite('test/test/_image.png', shrunken_img)

cv2.destroyAllWindows()
