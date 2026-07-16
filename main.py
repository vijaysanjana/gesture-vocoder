import cv2

from audio.vocoder_instrument import VocoderInstrument
from camera.webcam import Webcam
from config.settings import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    HAND_SMOOTHING,
)
from tracking.hand_tracker import HandTracker
from ui.overlay import Overlay


WINDOW_NAME = "GestureVox"


def main() -> None:
    webcam = Webcam(
        camera_index=CAMERA_INDEX,
        mirror=True,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    tracker = HandTracker(
        num_hands=1,
        smoothing_factor=HAND_SMOOTHING,
    )

    instrument = VocoderInstrument()
    overlay = Overlay()

    instrument.start()

    print("GestureVox vocoder started.")
    print("Pinch your fingers together for more robot voice.")
    print("Spread them apart for slightly more natural voice.")
    print("Press Q in the camera window to quit.")

    try:
        while True:
            frame = webcam.read()

            gesture_state = tracker.process(
                frame
            )

            instrument.update(
                gesture_state
            )

            wet_mix, carrier_frequency = (
                instrument.get_parameters()
            )

            output_frame = overlay.draw(
                frame=frame,
                gesture_state=gesture_state,
                frequency=carrier_frequency,
                volume=wet_mix,
            )

            cv2.imshow(
                WINDOW_NAME,
                output_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        instrument.stop()
        tracker.close()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()