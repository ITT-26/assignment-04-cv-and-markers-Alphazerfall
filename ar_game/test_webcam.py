import os

# Disabling the Windows Media Foundation hardware transforms allows the C920 to properly deliver MJPG frames at 1080p, which is essential for our AR game.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2

# 2. Switch from DirectShow (CAP_DSHOW) to Media Foundation (CAP_MSMF)
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

# 3. Request MJPG and 1080p
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    cv2.imshow("C920 MSMF Fix", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()