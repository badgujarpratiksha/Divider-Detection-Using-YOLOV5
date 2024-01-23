import cv2
import numpy as np

# Open the binary video file
video = cv2.VideoCapture('mask_output_video.avi')

# Get video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create VideoWriter object to save the processed video
output_video = cv2.VideoWriter('test_thinned_lines_output.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (frame_width, frame_height))

# Process each frame of the video
while True:
    ret, frame = video.read()

    if not ret:
        break

    # Convert frame to grayscale if needed
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define a larger kernel for erosion to decrease line width
    kernel_size = 15  # Increase kernel size for greater erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform erosion on the frame
    thinned_frame = cv2.erode(gray_frame, kernel, iterations=1)

    # Write the thinned frame to the output video
    output_video.write(cv2.cvtColor(thinned_frame, cv2.COLOR_GRAY2BGR))

    # Display the thinned frame (optional)
    cv2.imshow('Thinned Lines Video', thinned_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close windows
video.release()
output_video.release()
cv2.destroyAllWindows()
