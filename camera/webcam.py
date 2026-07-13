import cv2
import numpy as np


class Webcam:
    def __init__(
        self,
        camera_index: int = 0,
        mirror: bool = True,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.mirror = mirror

        self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera index {camera_index}."
            )

        if width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)

        if height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray:
        success, frame = self.capture.read()

        if not success:
            raise RuntimeError("Could not read a webcam frame.")

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return frame

    def release(self) -> None:
        if self.capture.isOpened():
            self.capture.release()