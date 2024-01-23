import cv2
import numpy as np

# Read the video file
cap = cv2.VideoCapture('test_Vid[2].avi')  # Replace with your AVI video file path

# Get video details - frame width, height, and fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object with AVI format
out = cv2.VideoWriter('mask_output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height), isColor=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine masks to get the final mask for red color
    mask = cv2.bitwise_or(mask1, mask2)

    # Set red areas to white and everything else to black in the frame
    binary_mask = np.zeros_like(frame)
    binary_mask[mask != 0] = [255, 255, 255]

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    # Write the processed frame to the output video file
    out.write(gray_frame)

cap.release()
out.release()
