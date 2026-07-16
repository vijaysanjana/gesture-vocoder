from __future__ import annotations

import numpy as np


class DelayLine:
    """A feedback delay line with a reusable circular buffer."""

    def __init__(
        self,
        delay_samples: int,
    ) -> None:
        if delay_samples <= 0:
            raise ValueError(
                "delay_samples must be positive."
            )

        self.buffer = np.zeros(
            delay_samples,
            dtype=np.float32,
        )

        self.position = 0

    def process(
        self,
        signal: np.ndarray,
        feedback: float,
    ) -> np.ndarray:
        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        frames = len(signal)

        if frames > len(self.buffer):
            raise ValueError(
                "Audio block is larger than the delay buffer."
            )

        delayed = self._read(
            frames
        )

        # Soft saturation keeps the feedback network stable.
        feedback_signal = np.tanh(
            signal + feedback * delayed
        ).astype(np.float32)

        self._write(
            feedback_signal
        )

        return delayed

    def _read(
        self,
        frames: int,
    ) -> np.ndarray:
        end = self.position + frames

        if end <= len(self.buffer):
            return self.buffer[
                self.position:end
            ].copy()

        first_length = (
            len(self.buffer)
            - self.position
        )

        second_length = (
            frames - first_length
        )

        return np.concatenate(
            (
                self.buffer[
                    self.position:
                ],
                self.buffer[
                    :second_length
                ],
            )
        )

    def _write(
        self,
        signal: np.ndarray,
    ) -> None:
        frames = len(signal)
        end = self.position + frames

        if end <= len(self.buffer):
            self.buffer[
                self.position:end
            ] = signal
        else:
            first_length = (
                len(self.buffer)
                - self.position
            )

            second_length = (
                frames - first_length
            )

            self.buffer[
                self.position:
            ] = signal[
                :first_length
            ]

            self.buffer[
                :second_length
            ] = signal[
                first_length:
            ]

        self.position = (
            self.position + frames
        ) % len(self.buffer)


class StereoReverb:
    """
    Lightweight stereo feedback-delay reverb.

    Different left and right delay times create width without requiring
    heavy convolution processing.
    """

    LEFT_DELAYS_MS = (
        31.0,
        43.0,
        59.0,
        71.0,
    )

    RIGHT_DELAYS_MS = (
        37.0,
        47.0,
        61.0,
        79.0,
    )

    def __init__(
        self,
        sample_rate: int,
        mix: float = 0.14,
        room_size: float = 0.68,
        stereo_width: float = 0.18,
    ) -> None:
        self.sample_rate = sample_rate

        self.mix = self._clamp(
            mix
        )

        self.room_size = self._clamp(
            room_size
        )

        self.stereo_width = self._clamp(
            stereo_width
        )

        self.left_delays = [
            DelayLine(
                self._milliseconds_to_samples(
                    milliseconds
                )
            )
            for milliseconds in self.LEFT_DELAYS_MS
        ]

        self.right_delays = [
            DelayLine(
                self._milliseconds_to_samples(
                    milliseconds
                )
            )
            for milliseconds in self.RIGHT_DELAYS_MS
        ]

    def process(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Process a stereo block with shape (frames, 2).
        """
        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        if signal.ndim != 2 or signal.shape[1] != 2:
            raise ValueError(
                "StereoReverb expects shape (frames, 2)."
            )

        if self.mix <= 0.0:
            return signal

        left = signal[:, 0]
        right = signal[:, 1]

        # Small crossfeed makes reflections spread across the stereo field.
        left_input = (
            left
            + self.stereo_width * right
        )

        right_input = (
            right
            + self.stereo_width * left
        )

        wet_left = np.zeros_like(
            left
        )

        wet_right = np.zeros_like(
            right
        )

        # Keep the feedback below 1.0 for stability.
        feedback = (
            0.42
            + self.room_size * 0.36
        )

        for delay in self.left_delays:
            wet_left += delay.process(
                signal=left_input,
                feedback=feedback,
            )

        for delay in self.right_delays:
            wet_right += delay.process(
                signal=right_input,
                feedback=feedback,
            )

        normalization = 1.0 / np.sqrt(
            len(self.left_delays)
        )

        wet_left *= normalization
        wet_right *= normalization

        wet = np.column_stack(
            (
                wet_left,
                wet_right,
            )
        ).astype(np.float32)

        dry_gain = np.sqrt(
            1.0 - self.mix
        )

        wet_gain = np.sqrt(
            self.mix
        )

        return (
            dry_gain * signal
            + wet_gain * wet
        ).astype(np.float32)

    def _milliseconds_to_samples(
        self,
        milliseconds: float,
    ) -> int:
        return max(
            1,
            int(
                milliseconds
                * self.sample_rate
                / 1000.0
            ),
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(1.0, value),
        )