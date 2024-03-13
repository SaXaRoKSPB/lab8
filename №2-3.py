import cv2
import numpy as np

video = cv2.VideoCapture(0)
down_points = (640, 480)

marker_image = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

    ret_tracker, thresh = cv2.threshold(gray, 110, 255,
                                        cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contours_video = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contours_video)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        marker_center_x = x + w // 2
        marker_center_y = y + h // 2

        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        distance = round(np.sqrt((marker_center_x - frame_center_x) ** 2 +
                                 (marker_center_y - frame_center_y) ** 2))

        cv2.putText(frame, f"Distance: {distance}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
