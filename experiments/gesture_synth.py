from pathlib import Path
import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd


CAMERA_INDEX = 1  # Change this if your MacBook camera uses another index.
MODEL_PATH = Path("models/hand_landmarker.task")

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_MCP = 9

SMOOTHING_FACTOR = 0.2

SAMPLE_RATE = 44_100
MIN_FREQUENCY = 130.81   # C3
MAX_FREQUENCY = 1046.50  # C6
MAX_VOLUME = 0.15

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class GestureSynth:
    def __init__(self) -> None:
        self.frequency = 440.0
        self.volume = 0.0
        self.phase = 0.0
        self.lock = threading.Lock()

        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=256,
        )

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()

    def update(self, frequency: float, volume: float) -> None:
        with self.lock:
            self.frequency = frequency
            self.volume = volume

    def _audio_callback(
        self,
        outdata,
        frames,
        time_info,
        status,
    ) -> None:
        if status:
            print(status)

        with self.lock:
            frequency = self.frequency
            volume = self.volume

        phase_increment = 2 * np.pi * frequency / SAMPLE_RATE

        phases = self.phase + phase_increment * np.arange(frames)

        samples = np.sin(phases) * volume

        self.phase = (
            phases[-1] + phase_increment
        ) % (2 * np.pi)

        outdata[:, 0] = samples.astype(np.float32)


def landmark_to_pixel(
    landmark,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    return (
        int(landmark.x * frame_width),
        int(landmark.y * frame_height),
    )


def distance_2d(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
) -> float:
    return math.hypot(
        point_b[0] - point_a[0],
        point_b[1] - point_a[1],
    )


def calculate_pinch(
    hand_landmarks,
    frame_width: int,
    frame_height: int,
) -> float:
    thumb = landmark_to_pixel(
        hand_landmarks[THUMB_TIP],
        frame_width,
        frame_height,
    )

    index = landmark_to_pixel(
        hand_landmarks[INDEX_TIP],
        frame_width,
        frame_height,
    )

    wrist = landmark_to_pixel(
        hand_landmarks[WRIST],
        frame_width,
        frame_height,
    )

    middle_mcp = landmark_to_pixel(
        hand_landmarks[MIDDLE_MCP],
        frame_width,
        frame_height,
    )

    pinch_distance = distance_2d(thumb, index)
    hand_size = distance_2d(wrist, middle_mcp)

    if hand_size == 0:
        return 0.0

    normalized_distance = pinch_distance / hand_size

    pinch_min = 0.15
    pinch_max = 1.2

    normalized_value = (
        normalized_distance - pinch_min
    ) / (
        pinch_max - pinch_min
    )

    return max(0.0, min(1.0, normalized_value))


def hand_height_to_frequency(hand_landmarks) -> float:
    wrist_y = hand_landmarks[WRIST].y

    # MediaPipe uses 0 at the top and 1 at the bottom.
    normalized_height = 1.0 - wrist_y
    normalized_height = max(0.0, min(1.0, normalized_height))

    # Exponential mapping creates more natural musical pitch spacing.
    return MIN_FREQUENCY * (
        MAX_FREQUENCY / MIN_FREQUENCY
    ) ** normalized_height


def draw_hand_landmarks(frame, hand_landmarks) -> None:
    frame_height, frame_width, _ = frame.shape

    points = [
        landmark_to_pixel(
            landmark,
            frame_width,
            frame_height,
        )
        for landmark in hand_landmarks
    ]

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            (255, 255, 255),
            2,
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            5,
            (0, 255, 0),
            -1,
        )


def draw_value_bar(
    frame,
    label: str,
    value: float,
    y_position: int,
) -> None:
    bar_x = 20
    bar_width = 300
    bar_height = 25

    value = max(0.0, min(1.0, value))
    filled_width = int(bar_width * value)

    cv2.putText(
        frame,
        f"{label}: {value:.2f}",
        (bar_x, y_position - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.rectangle(
        frame,
        (bar_x, y_position),
        (bar_x + bar_width, y_position + bar_height),
        (255, 255, 255),
        2,
    )

    cv2.rectangle(
        frame,
        (bar_x, y_position),
        (bar_x + filled_width, y_position + bar_height),
        (255, 0, 255),
        -1,
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run this script from the repo root."
        )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}."
        )

    synth = GestureSynth()
    synth.start()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start_time = time.perf_counter()

    smoothed_pinch = 0.0
    smoothed_frequency = 440.0

    print("Gesture synth started. Press Q to quit.")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a camera frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                frame_height, frame_width, _ = frame.shape

                pinch_value = None
                frequency = None
                volume = 0.0

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]

                    draw_hand_landmarks(
                        frame,
                        hand_landmarks,
                    )

                    raw_pinch = calculate_pinch(
                        hand_landmarks,
                        frame_width,
                        frame_height,
                    )

                    smoothed_pinch = (
                        SMOOTHING_FACTOR * raw_pinch
                        + (1 - SMOOTHING_FACTOR) * smoothed_pinch
                    )

                    pinch_value = smoothed_pinch

                    raw_frequency = hand_height_to_frequency(
                        hand_landmarks
                    )

                    smoothed_frequency = (
                        SMOOTHING_FACTOR * raw_frequency
                        + (1 - SMOOTHING_FACTOR) * smoothed_frequency
                    )

                    frequency = smoothed_frequency
                    volume = pinch_value * MAX_VOLUME

                    synth.update(
                        frequency=frequency,
                        volume=volume,
                    )

                    thumb_point = landmark_to_pixel(
                        hand_landmarks[THUMB_TIP],
                        frame_width,
                        frame_height,
                    )

                    index_point = landmark_to_pixel(
                        hand_landmarks[INDEX_TIP],
                        frame_width,
                        frame_height,
                    )

                    cv2.line(
                        frame,
                        thumb_point,
                        index_point,
                        (255, 0, 255),
                        3,
                    )

                else:
                    synth.update(
                        frequency=440.0,
                        volume=0.0,
                    )

                cv2.putText(
                    frame,
                    f"Hands: {len(result.hand_landmarks)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                if pinch_value is not None:
                    draw_value_bar(
                        frame,
                        label="Pinch / Volume",
                        value=pinch_value,
                        y_position=90,
                    )

                if frequency is not None:
                    cv2.putText(
                        frame,
                        f"Frequency: {frequency:.1f} Hz",
                        (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Volume: {volume:.3f}",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(
                    "Gesture Vocoder - Gesture Synth",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        synth.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()