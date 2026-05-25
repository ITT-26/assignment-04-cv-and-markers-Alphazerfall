import cv2
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print("FOURCC:", "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)]))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Size:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("ID 1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()