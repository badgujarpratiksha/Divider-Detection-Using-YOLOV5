import cv2
import numpy as np

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary_mask, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None and len(lines) > 0:
        # Separate positive and negative slope lines
        positive_lines = []
        negative_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                if slope > 0:
                    positive_lines.append(line)
                else:
                    negative_lines.append(line)

        # Calculate average positive and negative slope lines
        if positive_lines:
            avg_positive_line = np.mean(positive_lines, axis=0, dtype=np.int32)
            cv2.line(frame, (avg_positive_line[0][0], avg_positive_line[0][1]),
                     (avg_positive_line[0][2], avg_positive_line[0][3]), (0, 0, 255), thickness=10)  # Red color

        if negative_lines:
            avg_negative_line = np.mean(negative_lines, axis=0, dtype=np.int32)
            cv2.line(frame, (avg_negative_line[0][0], avg_negative_line[0][1]),
                     (avg_negative_line[0][2], avg_negative_line[0][3]), (0, 0, 255), thickness=10)  # Red color

    return frame

video_capture = cv2.VideoCapture('test_thinned_lines_output.mp4')
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    processed_frame = process_frame(frame)

    # Write the processed frame to the output video
    out.write(processed_frame)

    cv2.imshow('Processed Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
