from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


Color = tuple[int, int, int]
Point = tuple[int, int]


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


class GlowCanvas:
    """
    Collects all glowing shapes on one shared layer.

    Draw every glow shape first, then call composite() once per frame.
    """

    def __init__(
        self,
        frame_shape: tuple[int, ...],
    ) -> None:
        self.layer = np.zeros(
            frame_shape,
            dtype=np.uint8,
        )

    def clear(self) -> None:
        self.layer.fill(0)

    def line(
        self,
        start: Point,
        end: Point,
        color: Color,
        thickness: int = 8,
    ) -> None:
        cv2.line(
            self.layer,
            start,
            end,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def circle(
        self,
        center: Point,
        radius: int,
        color: Color,
        thickness: int = -1,
    ) -> None:
        cv2.circle(
            self.layer,
            center,
            radius,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def polyline(
        self,
        points: Iterable[Point],
        color: Color,
        thickness: int = 8,
        closed: bool = False,
    ) -> None:
        point_list = list(points)

        if len(point_list) < 2:
            return

        contour = np.array(
            point_list,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            self.layer,
            [contour],
            closed,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def composite(
        self,
        frame: np.ndarray,
        blur_radius: float = 8.0,
        intensity: float = 0.6,
    ) -> None:
        """
        Blur the entire glow layer once and add it to the frame.
        """
        if intensity <= 0:
            return

        blurred = cv2.GaussianBlur(
            self.layer,
            (0, 0),
            sigmaX=blur_radius,
            sigmaY=blur_radius,
        )

        scaled = cv2.convertScaleAbs(
            blurred,
            alpha=intensity,
        )

        cv2.add(
            frame,
            scaled,
            dst=frame,
        )


def draw_line(
    frame: np.ndarray,
    start: Point,
    end: Point,
    color: Color,
    thickness: int = 2,
) -> None:
    cv2.line(
        frame,
        start,
        end,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_circle(
    frame: np.ndarray,
    center: Point,
    radius: int,
    color: Color,
    thickness: int = -1,
) -> None:
    cv2.circle(
        frame,
        center,
        radius,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_polyline(
    frame: np.ndarray,
    points: Iterable[Point],
    color: Color,
    thickness: int = 2,
    closed: bool = False,
) -> None:
    point_list = list(points)

    if len(point_list) < 2:
        return

    contour = np.array(
        point_list,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.polylines(
        frame,
        [contour],
        closed,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_translucent_panel(
    frame: np.ndarray,
    top_left: Point,
    bottom_right: Point,
    color: Color,
    opacity: float = 0.55,
    border_color: Color | None = None,
    border_thickness: int = 1,
) -> None:
    opacity = clamp(opacity)

    panel_layer = frame.copy()

    cv2.rectangle(
        panel_layer,
        top_left,
        bottom_right,
        color,
        -1,
        cv2.LINE_AA,
    )

    cv2.addWeighted(
        panel_layer,
        opacity,
        frame,
        1.0 - opacity,
        0,
        frame,
    )

    if border_color is not None:
        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            border_color,
            border_thickness,
            cv2.LINE_AA,
        )


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: Point,
    color: Color,
    scale: float = 0.7,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_meter(
    frame: np.ndarray,
    value: float,
    origin: Point,
    width: int,
    height: int,
    foreground: Color,
    background: Color,
    border: Color | None = None,
) -> None:
    value = clamp(value)

    x, y = origin
    filled_width = int(width * value)

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        background,
        -1,
        cv2.LINE_AA,
    )

    if filled_width > 0:
        cv2.rectangle(
            frame,
            (x, y),
            (x + filled_width, y + height),
            foreground,
            -1,
            cv2.LINE_AA,
        )

    if border is not None:
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            border,
            1,
            cv2.LINE_AA,
        )


def make_energy_beam_points(
    start: Point,
    end: Point,
    intensity: float,
    segments: int = 20,
    amplitude: float = 5.0,
    phase: float = 0.0,
) -> list[Point]:
    """
    Return the points for a wavy energy beam.

    Drawing is handled separately so the same points can be sent to both
    the glow canvas and the crisp foreground layer.
    """
    intensity = clamp(intensity)

    start_array = np.array(
        start,
        dtype=np.float32,
    )

    end_array = np.array(
        end,
        dtype=np.float32,
    )

    direction = end_array - start_array
    length = float(np.linalg.norm(direction))

    if length < 1.0:
        return []

    perpendicular = np.array(
        [-direction[1], direction[0]],
        dtype=np.float32,
    ) / length

    points: list[Point] = []

    for index in range(segments + 1):
        t = index / segments

        position = start_array + direction * t

        wave = np.sin(
            t * np.pi * 4.0 + phase
        )

        offset = (
            perpendicular
            * wave
            * amplitude
            * intensity
        )

        point = position + offset

        points.append(
            (
                int(point[0]),
                int(point[1]),
            )
        )

    return points