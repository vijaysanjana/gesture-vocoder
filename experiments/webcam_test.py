import cv2


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print("Camera started. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the webcam.")
                break

            # Mirror the video so movement feels natural
            frame = cv2.flip(frame, 1)

            cv2.imshow("Gesture Vocoder — Webcam Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = Esc key 
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()