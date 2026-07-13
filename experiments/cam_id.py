import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(f"Camera {i} works!")

        ret, frame = cap.read()

        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(0)

        cap.release()

cv2.destroyAllWindows()