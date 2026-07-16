from __future__ import annotations

import numpy as np


class CarrierSynth:
    """
    Generates one harmonically rich carrier note.

    Harmony is created by running multiple CarrierSynth and Vocoder
    instances in parallel, rather than placing an entire chord into one
    carrier signal.
    """

    def __init__(
        self,
        sample_rate: int,
        frequency: float = 196.0,
        detune_ratio: float = 0.0025,
        noise_amount: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.detune_ratio = detune_ratio
        self.noise_amount = noise_amount

        self.lower_phase = 0.0
        self.center_phase = 0.0
        self.upper_phase = 0.0

        self.random = np.random.default_rng()

    def set_frequency(
        self,
        frequency: float,
    ) -> None:
        self.frequency = max(20.0, frequency)

    def generate(
        self,
        frames: int,
    ) -> np.ndarray:
        lower, self.lower_phase = self._generate_wave(
            phase=self.lower_phase,
            frequency=self.frequency * (1.0 - self.detune_ratio),
            frames=frames,
        )

        center, self.center_phase = self._generate_wave(
            phase=self.center_phase,
            frequency=self.frequency,
            frames=frames,
        )

        upper, self.upper_phase = self._generate_wave(
            phase=self.upper_phase,
            frequency=self.frequency * (1.0 + self.detune_ratio),
            frames=frames,
        )

        carrier = (
            0.25 * lower
            + 0.50 * center
            + 0.25 * upper
        )

        if self.noise_amount > 0.0:
            carrier += (
                self.random.standard_normal(frames)
                * self.noise_amount
            )

        # Gentle saturation controls peaks.
        carrier = np.tanh(carrier * 1.2)

        return carrier.astype(np.float32)

    def _generate_wave(
        self,
        phase: float,
        frequency: float,
        frames: int,
    ) -> tuple[np.ndarray, float]:
        increment = frequency / self.sample_rate

        phases = (
            phase
            + increment * np.arange(frames)
        )

        wrapped = phases % 1.0

        saw = 2.0 * (wrapped - 0.5)

        sine = np.sin(
            2.0 * np.pi * wrapped
        )

        # Keep enough saw harmonics for speech intelligibility while
        # softening the harsh buzz with a sine component.
        wave = (
            0.72 * saw
            + 0.28 * sine
        )

        new_phase = (
            phases[-1] + increment
        ) % 1.0

        return wave, new_phase