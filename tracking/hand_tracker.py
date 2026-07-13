from pathlib import Path
import math
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from gestures.gesture_state import GestureState, HandState


THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_MCP = 9


class HandTracker:
    def __init__(
        self,
        num_hands: int = 2,
        smoothing_factor: float = 0.2,
        model_path: Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]

        self.model_path = (
            model_path
            if model_path is not None
            else project_root / "models" / "hand_landmarker.task"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found at {self.model_path}."
            )

        self.smoothing_factor = smoothing_factor
        self.start_time = time.perf_counter()

        self.smoothed_values: dict[str, dict[str, float]] = {
            "Left": {
                "pinch": 0.0,
                "height": 0.5,
            },
            "Right": {
                "pinch": 0.0,
                "height": 0.5,
            },
        }

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(
                options
            )
        )

    def process(self, frame: np.ndarray) -> GestureState:
        """
        Detect hands in an OpenCV BGR frame and return normalized state.
        """
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int(
            (time.perf_counter() - self.start_time) * 1000
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        state = GestureState()

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):
            handedness = self._get_handedness(
                result,
                hand_index,
            )

            raw_pinch = self._calculate_pinch(
                landmarks,
                frame.shape[1],
                frame.shape[0],
            )

            raw_height = self._calculate_height(
                landmarks
            )

            pinch = self._smooth(
                handedness,
                "pinch",
                raw_pinch,
            )

            height = self._smooth(
                handedness,
                "height",
                raw_height,
            )

            hand_state = HandState(
                pinch=pinch,
                height=height,
                handedness=handedness,
                landmarks=list(landmarks),
            )

            if handedness == "Left":
                state.left = hand_state
            else:
                state.right = hand_state

        return state

    def close(self) -> None:
        self.landmarker.close()

    def _get_handedness(
        self,
        result: Any,
        hand_index: int,
    ) -> str:
        try:
            category = result.handedness[hand_index][0]
            name = category.category_name

            if name in {"Left", "Right"}:
                return name
        except (IndexError, AttributeError):
            pass

        return "Right"

    def _calculate_pinch(
        self,
        landmarks: list[Any],
        frame_width: int,
        frame_height: int,
    ) -> float:
        thumb = self._landmark_to_pixel(
            landmarks[THUMB_TIP],
            frame_width,
            frame_height,
        )

        index = self._landmark_to_pixel(
            landmarks[INDEX_TIP],
            frame_width,
            frame_height,
        )

        wrist = self._landmark_to_pixel(
            landmarks[WRIST],
            frame_width,
            frame_height,
        )

        middle_mcp = self._landmark_to_pixel(
            landmarks[MIDDLE_MCP],
            frame_width,
            frame_height,
        )

        pinch_distance = self._distance(
            thumb,
            index,
        )

        hand_size = self._distance(
            wrist,
            middle_mcp,
        )

        if hand_size == 0:
            return 0.0

        normalized_distance = (
            pinch_distance / hand_size
        )

        pinch_min = 0.15
        pinch_max = 1.2

        normalized_pinch = (
            normalized_distance - pinch_min
        ) / (
            pinch_max - pinch_min
        )

        return self._clamp(normalized_pinch)

    def _calculate_height(
        self,
        landmarks: list[Any],
    ) -> float:
        wrist_y = landmarks[WRIST].y

        # MediaPipe y=0 is the top of the frame.
        return self._clamp(1.0 - wrist_y)

    def _smooth(
        self,
        handedness: str,
        value_name: str,
        new_value: float,
    ) -> float:
        previous_value = self.smoothed_values[
            handedness
        ][value_name]

        smoothed_value = (
            self.smoothing_factor * new_value
            + (1.0 - self.smoothing_factor)
            * previous_value
        )

        self.smoothed_values[
            handedness
        ][value_name] = smoothed_value

        return smoothed_value

    @staticmethod
    def _landmark_to_pixel(
        landmark: Any,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        return (
            int(landmark.x * frame_width),
            int(landmark.y * frame_height),
        )

    @staticmethod
    def _distance(
        point_a: tuple[int, int],
        point_b: tuple[int, int],
    ) -> float:
        return math.hypot(
            point_b[0] - point_a[0],
            point_b[1] - point_a[1],
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))