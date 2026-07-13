from pathlib import Path
import math
import time

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
MODEL_PATH = Path("models/hand_landmarker.task")

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_MCP = 9

SMOOTHING_FACTOR = 0.2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                                # Palm edge
]


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


def draw_pinch_bar(
    frame,
    pinch_value: float,
) -> None:
    bar_x = 20
    bar_y = 80
    bar_width = 300
    bar_height = 25

    filled_width = int(bar_width * pinch_value)

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255),
        2,
    )

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + filled_width, bar_y + bar_height),
        (255, 0, 255),
        -1,
    )

    cv2.putText(
        frame,
        f"Pinch: {pinch_value:.2f}",
        (bar_x, bar_y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run this script from the repo root and make sure the model exists."
        )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}."
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start_time = time.perf_counter()
    smoothed_pinch = 0.0

    print("Pinch tracking started. Press Q to quit.")

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

                for hand_landmarks in result.hand_landmarks:
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

                hand_count = len(result.hand_landmarks)

                cv2.putText(
                    frame,
                    f"Hands: {hand_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                if pinch_value is not None:
                    draw_pinch_bar(
                        frame,
                        pinch_value,
                    )

                cv2.imshow(
                    "Gesture Vocoder - Pinch Tracking",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()