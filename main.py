import cv2

from audio.synth import Synth
from camera.webcam import Webcam
from tracking.hand_tracker import HandTracker
from ui.overlay import Overlay


CAMERA_INDEX = 0
WINDOW_NAME = "Gesture Vocoder"


def main() -> None:
    webcam = Webcam(
        camera_index=CAMERA_INDEX,
        mirror=True,
    )

    tracker = HandTracker(
        num_hands=2,
        smoothing_factor=0.2,
    )

    synth = Synth(
        waveform="sine",
    )

    overlay = Overlay()

    synth.start()

    print("Gesture Vocoder started.")
    print("Press Q in the camera window to quit.")

    try:
        while True:
            frame = webcam.read()

            gesture_state = tracker.process(
                frame
            )

            synth.update(
                gesture_state
            )

            frequency, volume = (
                synth.get_parameters()
            )

            output_frame = overlay.draw(
                frame=frame,
                gesture_state=gesture_state,
                frequency=frequency,
                volume=volume,
            )

            cv2.imshow(
                WINDOW_NAME,
                output_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        synth.stop()
        tracker.close()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()