import cv2
import numpy as np

def draw_hough_lines_on_mask(video):
    output_video = []
    for frame in video:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw lines on the mask
        
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)  # Combine original frame with line mask
        output_video.append(combo_image)
    return output_video

# Open the binary video file
video = cv2.VideoCapture('test_thinned_lines_output.mp4')

# Store video frames in a list
frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    frames.append(frame)
    del frame  # Release frame explicitly

# Draw Hough lines on the mask for video frames
output_frames = draw_hough_lines_on_mask(frames)

# Release individual frames from memory
for frame in frames:
    del frame

# Save the output video
output_video = cv2.VideoWriter('output_video_with_hough_lines.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               30, (frames[0].shape[1], frames[0].shape[0]))

for frame in output_frames:
    output_video.write(frame)

# Release video objects and close windows
video.release()
output_video.release()
cv2.destroyAllWindows()
