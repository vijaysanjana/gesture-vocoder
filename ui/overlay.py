from typing import Any

import cv2
import numpy as np

from gestures.gesture_state import GestureState, HandState


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


THUMB_TIP = 4
INDEX_TIP = 8


class Overlay:
    def draw(
        self,
        frame: np.ndarray,
        gesture_state: GestureState,
        frequency: float,
        volume: float,
    ) -> np.ndarray:
        for hand in [
            gesture_state.left,
            gesture_state.right,
        ]:
            if hand is not None:
                self._draw_hand(
                    frame,
                    hand,
                )

        self._draw_status(
            frame,
            gesture_state,
            frequency,
            volume,
        )

        return frame

    def _draw_hand(
        self,
        frame: np.ndarray,
        hand: HandState,
    ) -> None:
        frame_height, frame_width, _ = frame.shape

        points = [
            self._landmark_to_pixel(
                landmark,
                frame_width,
                frame_height,
            )
            for landmark in hand.landmarks
        ]

        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start_index],
                points[end_index],
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for point in points:
            cv2.circle(
                frame,
                point,
                5,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

        cv2.line(
            frame,
            points[THUMB_TIP],
            points[INDEX_TIP],
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )

    def _draw_status(
        self,
        frame: np.ndarray,
        gesture_state: GestureState,
        frequency: float,
        volume: float,
    ) -> None:
        hand = gesture_state.primary_hand

        hand_count = int(
            gesture_state.left is not None
        ) + int(
            gesture_state.right is not None
        )

        cv2.putText(
            frame,
            f"Hands: {hand_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if hand is None:
            cv2.putText(
                frame,
                "Show your hand to begin",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return

        self._draw_value_bar(
            frame=frame,
            label="Pinch / Volume",
            value=hand.pinch,
            y_position=90,
        )

        cv2.putText(
            frame,
            f"Frequency: {frequency:.1f} Hz",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Volume: {volume:.3f}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_value_bar(
        self,
        frame: np.ndarray,
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
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            frame,
            (bar_x, y_position),
            (
                bar_x + bar_width,
                y_position + bar_height,
            ),
            (255, 255, 255),
            2,
        )

        cv2.rectangle(
            frame,
            (bar_x, y_position),
            (
                bar_x + filled_width,
                y_position + bar_height,
            ),
            (255, 0, 255),
            -1,
        )

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