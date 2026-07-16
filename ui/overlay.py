from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from gestures.gesture_state import GestureState, HandState
from ui.animation import AnimatedValue
from ui.drawing import (
    GlowCanvas,
    draw_circle,
    draw_line,
    draw_meter,
    draw_polyline,
    draw_text,
    draw_translucent_panel,
    make_energy_beam_points,
)
from ui.theme import Theme


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

FINGERTIPS = [4, 8, 12, 16, 20]

THUMB_TIP = 4
INDEX_TIP = 8


class Overlay:
    def __init__(
        self,
        theme: Theme | None = None,
    ) -> None:
        self.theme = theme or Theme()

        # Raw pinch is still useful for visualizing the physical gesture.
        self.display_pinch = AnimatedValue(
            initial=0.0,
            speed=0.16,
        )

        # This represents the actual dry/wet vocoder mix.
        self.display_wet_mix = AnimatedValue(
            initial=0.95,
            speed=0.14,
        )

        self.hand_visibility = AnimatedValue(
            initial=0.0,
            speed=0.18,
        )

        self.beam_intensity = AnimatedValue(
            initial=0.0,
            speed=0.20,
        )

        self.start_time = time.perf_counter()

        self.glow_canvas: GlowCanvas | None = None
        self.glow_shape: tuple[int, ...] | None = None

    def draw(
        self,
        frame: np.ndarray,
        gesture_state: GestureState,
        frequency: float,
        volume: float,
    ) -> np.ndarray:
        """
        Render the gesture-controlled vocoder interface.

        `frequency` is the fixed carrier frequency.
        `volume` currently contains the vocoder wet-mix value supplied
        by VocoderInstrument.
        """
        output = frame.copy()
        wet_mix = self._clamp(volume)

        self._ensure_glow_canvas(output)
        assert self.glow_canvas is not None

        self.glow_canvas.clear()

        hand = gesture_state.primary_hand

        self._update_animation_targets(
            hand=hand,
            wet_mix=wet_mix,
        )

        display_pinch = self.display_pinch.update()
        display_wet_mix = self.display_wet_mix.update()
        visibility = self.hand_visibility.update()
        beam_intensity = self.beam_intensity.update()

        self._apply_background_tint(output)

        hand_render_data: list[
            tuple[HandState, list[tuple[int, int]]]
        ] = []

        for detected_hand in [
            gesture_state.left,
            gesture_state.right,
        ]:
            if detected_hand is None:
                continue

            points = self._hand_points(
                frame=output,
                hand=detected_hand,
            )

            hand_render_data.append(
                (detected_hand, points)
            )

            self._draw_hand_glow(
                glow=self.glow_canvas,
                points=points,
                visibility=visibility,
                beam_intensity=beam_intensity,
            )

        # Composite all glow with one blur for good performance.
        self.glow_canvas.composite(
            frame=output,
            blur_radius=7.0,
            intensity=0.75,
        )

        # Draw crisp shapes after adding the glow.
        for detected_hand, points in hand_render_data:
            self._draw_hand_foreground(
                frame=output,
                hand=detected_hand,
                points=points,
                visibility=visibility,
                beam_intensity=beam_intensity,
            )

        self._draw_hud(
            frame=output,
            hand=hand,
            display_pinch=display_pinch,
            display_wet_mix=display_wet_mix,
            frequency=frequency,
        )

        return output

    def _ensure_glow_canvas(
        self,
        frame: np.ndarray,
    ) -> None:
        if (
            self.glow_canvas is None
            or self.glow_shape != frame.shape
        ):
            self.glow_canvas = GlowCanvas(
                frame.shape
            )
            self.glow_shape = frame.shape

    def _update_animation_targets(
        self,
        hand: HandState | None,
        wet_mix: float,
    ) -> None:
        # Wet mix comes from the audio instrument, so always animate toward it.
        self.display_wet_mix.set_target(
            self._clamp(wet_mix)
        )

        if hand is None:
            self.hand_visibility.set_target(0.0)
            self.beam_intensity.set_target(0.0)
            return

        self.hand_visibility.set_target(1.0)
        self.display_pinch.set_target(hand.pinch)

        # Fingers together means stronger robotic processing.
        closeness = 1.0 - hand.pinch

        self.beam_intensity.set_target(
            self._clamp(closeness)
        )

    def _apply_background_tint(
        self,
        frame: np.ndarray,
    ) -> None:
        tint = np.full_like(
            frame,
            self.theme.background_tint,
        )

        cv2.addWeighted(
            tint,
            0.06,
            frame,
            0.94,
            0,
            frame,
        )

    def _hand_points(
        self,
        frame: np.ndarray,
        hand: HandState,
    ) -> list[tuple[int, int]]:
        frame_height, frame_width, _ = frame.shape

        return [
            self._landmark_to_pixel(
                landmark,
                frame_width,
                frame_height,
            )
            for landmark in hand.landmarks
        ]

    def _draw_hand_glow(
        self,
        glow: GlowCanvas,
        points: list[tuple[int, int]],
        visibility: float,
        beam_intensity: float,
    ) -> None:
        skeleton_color = self._scale_color(
            self.theme.skeleton,
            visibility,
        )

        fingertip_color = self._scale_color(
            self.theme.fingertip,
            visibility,
        )

        for start_index, end_index in HAND_CONNECTIONS:
            glow.line(
                start=points[start_index],
                end=points[end_index],
                color=skeleton_color,
                thickness=8,
            )

        pulse = self._pulse(
            speed=3.0,
            minimum=0.85,
            maximum=1.15,
        )

        for index in FINGERTIPS:
            glow.circle(
                center=points[index],
                radius=max(
                    6,
                    int(10 * pulse),
                ),
                color=fingertip_color,
                thickness=-1,
            )

        beam_points = self._beam_points(
            points=points,
            beam_intensity=beam_intensity,
        )

        if beam_points:
            glow.polyline(
                points=beam_points,
                color=self.theme.pinch,
                thickness=max(
                    6,
                    int(7 + beam_intensity * 5),
                ),
            )

        if beam_intensity > 0.05:
            glow.circle(
                center=self._pinch_midpoint(points),
                radius=int(
                    10 + beam_intensity * 18
                ),
                color=self.theme.pinch,
                thickness=-1,
            )

    def _draw_hand_foreground(
        self,
        frame: np.ndarray,
        hand: HandState,
        points: list[tuple[int, int]],
        visibility: float,
        beam_intensity: float,
    ) -> None:
        skeleton_color = self._scale_color(
            self.theme.skeleton,
            visibility,
        )

        fingertip_color = self._scale_color(
            self.theme.fingertip,
            visibility,
        )

        for start_index, end_index in HAND_CONNECTIONS:
            draw_line(
                frame=frame,
                start=points[start_index],
                end=points[end_index],
                color=skeleton_color,
                thickness=2,
            )

        pulse = self._pulse(
            speed=3.0,
            minimum=0.85,
            maximum=1.15,
        )

        for index in FINGERTIPS:
            draw_circle(
                frame=frame,
                center=points[index],
                radius=max(
                    3,
                    int(5 * pulse),
                ),
                color=fingertip_color,
                thickness=-1,
            )

        beam_points = self._beam_points(
            points=points,
            beam_intensity=beam_intensity,
        )

        if beam_points:
            draw_polyline(
                frame=frame,
                points=beam_points,
                color=self.theme.pinch,
                thickness=max(
                    1,
                    int(1 + beam_intensity * 2),
                ),
            )

        if beam_intensity > 0.05:
            pulse = self._pulse(
                speed=5.0,
                minimum=0.8,
                maximum=1.2,
            )

            radius = int(
                (
                    7
                    + 14 * beam_intensity
                )
                * pulse
            )

            draw_circle(
                frame=frame,
                center=self._pinch_midpoint(points),
                radius=max(4, radius),
                color=self.theme.pinch,
                thickness=2,
            )

    def _beam_points(
        self,
        points: list[tuple[int, int]],
        beam_intensity: float,
    ) -> list[tuple[int, int]]:
        phase = (
            time.perf_counter()
            - self.start_time
        ) * 7.0

        return make_energy_beam_points(
            start=points[THUMB_TIP],
            end=points[INDEX_TIP],
            intensity=beam_intensity,
            segments=18,
            amplitude=2.0 + 5.0 * beam_intensity,
            phase=phase,
        )

    @staticmethod
    def _pinch_midpoint(
        points: list[tuple[int, int]],
    ) -> tuple[int, int]:
        thumb = points[THUMB_TIP]
        index = points[INDEX_TIP]

        return (
            (thumb[0] + index[0]) // 2,
            (thumb[1] + index[1]) // 2,
        )

    def _draw_hud(
        self,
        frame: np.ndarray,
        hand: HandState | None,
        display_pinch: float,
        display_wet_mix: float,
        frequency: float,
    ) -> None:
        _, frame_width, _ = frame.shape

        panel_x = 24
        panel_y = 24
        panel_width = min(
            320,
            frame_width - 48,
        )
        panel_height = 128

        draw_translucent_panel(
            frame=frame,
            top_left=(
                panel_x,
                panel_y,
            ),
            bottom_right=(
                panel_x + panel_width,
                panel_y + panel_height,
            ),
            color=self.theme.panel,
            opacity=0.48,
            border_color=self.theme.panel_border,
        )

        draw_text(
            frame=frame,
            text="GESTUREVOX",
            origin=(
                panel_x + 16,
                panel_y + 27,
            ),
            color=self.theme.text,
            scale=0.68,
            thickness=2,
        )

        carrier_note = self._frequency_to_note(
            frequency
        )

        # Show carrier information even when tracking briefly drops.
        draw_text(
            frame=frame,
            text="CARRIER",
            origin=(
                panel_x + 16,
                panel_y + 56,
            ),
            color=self.theme.muted_text,
            scale=0.42,
        )

        draw_text(
            frame=frame,
            text=carrier_note,
            origin=(
                panel_x + 16,
                panel_y + 79,
            ),
            color=self.theme.text,
            scale=0.82,
            thickness=2,
        )

        draw_text(
            frame=frame,
            text="ROBOT MIX",
            origin=(
                panel_x + 88,
                panel_y + 56,
            ),
            color=self.theme.muted_text,
            scale=0.42,
        )

        draw_meter(
            frame=frame,
            value=display_wet_mix,
            origin=(
                panel_x + 88,
                panel_y + 68,
            ),
            width=200,
            height=8,
            foreground=self.theme.pinch,
            background=self.theme.panel_border,
        )

        mix_percentage = round(
            display_wet_mix * 100
        )

        draw_text(
            frame=frame,
            text=f"{mix_percentage}%",
            origin=(
                panel_x + 236,
                panel_y + 101,
            ),
            color=self.theme.text,
            scale=0.48,
            thickness=1,
        )

        if hand is None:
            draw_text(
                frame=frame,
                text="SHOW HAND TO CONTROL MIX",
                origin=(
                    panel_x + 16,
                    panel_y + 112,
                ),
                color=self.theme.muted_text,
                scale=0.42,
            )
            return

        # A subtle physical-gesture indicator beneath the main mix display.
        draw_text(
            frame=frame,
            text="PINCH",
            origin=(
                panel_x + 16,
                panel_y + 106,
            ),
            color=self.theme.muted_text,
            scale=0.36,
        )

        draw_meter(
            frame=frame,
            value=1.0 - display_pinch,
            origin=(
                panel_x + 62,
                panel_y + 99,
            ),
            width=150,
            height=5,
            foreground=self.theme.skeleton,
            background=self.theme.panel_border,
        )

    @staticmethod
    def _frequency_to_note(
        frequency: float,
    ) -> str:
        if frequency <= 0:
            return "--"

        note_names = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]

        midi_note = round(
            69
            + 12
            * math.log2(
                frequency / 440.0
            )
        )

        note_name = note_names[
            midi_note % 12
        ]

        octave = midi_note // 12 - 1

        return f"{note_name}{octave}"

    @staticmethod
    def _landmark_to_pixel(
        landmark: Any,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        return (
            int(
                landmark.x
                * frame_width
            ),
            int(
                landmark.y
                * frame_height
            ),
        )

    @staticmethod
    def _scale_color(
        color: tuple[int, int, int],
        strength: float,
    ) -> tuple[int, int, int]:
        strength = Overlay._clamp(
            strength
        )

        return tuple(
            int(channel * strength)
            for channel in color
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        return max(
            minimum,
            min(maximum, value),
        )

    def _pulse(
        self,
        speed: float,
        minimum: float,
        maximum: float,
    ) -> float:
        elapsed = (
            time.perf_counter()
            - self.start_time
        )

        wave = (
            math.sin(
                elapsed * speed
            )
            + 1.0
        ) / 2.0

        return (
            minimum
            + wave
            * (
                maximum
                - minimum
            )
        )