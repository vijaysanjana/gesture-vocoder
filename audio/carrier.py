from __future__ import annotations

import numpy as np


class CarrierSynth:
    """
    Generates a harmonically rich chord carrier for an airy vocoder sound.

    Each chord tone uses two slightly detuned oscillators:
    root, third, fifth, and octave.
    """

    CHORD_INTERVALS = {
        "major": (0,),
        "minor": (0,),
        "sus2": (0,),
        "sus4": (0,),
    }

    def __init__(
        self,
        sample_rate: int,
        frequency: float = 130.81,
        chord_type: str = "major",
        noise_amount: float = 0.00,
    ) -> None:
        if chord_type not in self.CHORD_INTERVALS:
            raise ValueError(
                f"Unsupported chord type: {chord_type}"
            )

        self.sample_rate = sample_rate
        self.frequency = frequency
        self.chord_type = chord_type
        self.noise_amount = noise_amount

        self.detune_ratio = 0.0025
        self.random = np.random.default_rng()

        # Two oscillator phases for each of the four chord tones.
        self.phases = np.zeros(
            (1, 2),
            dtype=np.float64,
        )

    def set_frequency(
        self,
        frequency: float,
    ) -> None:
        self.frequency = max(20.0, frequency)

    def set_chord_type(
        self,
        chord_type: str,
    ) -> None:
        if chord_type not in self.CHORD_INTERVALS:
            raise ValueError(
                f"Unsupported chord type: {chord_type}"
            )

        self.chord_type = chord_type

    def generate(
        self,
        frames: int,
    ) -> np.ndarray:
        chord_frequencies = self._get_chord_frequencies()

        output = np.zeros(
            frames,
            dtype=np.float64,
        )

        # Lower chord tones should carry slightly more weight.
        chord_weights = (
            1.0,
        )

        for tone_index, tone_frequency in enumerate(
            chord_frequencies
        ):
            lower_frequency = (
                tone_frequency
                * (1.0 - self.detune_ratio)
            )

            upper_frequency = (
                tone_frequency
                * (1.0 + self.detune_ratio)
            )

            lower_wave, self.phases[tone_index, 0] = (
                self._generate_soft_wave(
                    phase=self.phases[tone_index, 0],
                    frequency=lower_frequency,
                    frames=frames,
                )
            )

            upper_wave, self.phases[tone_index, 1] = (
                self._generate_soft_wave(
                    phase=self.phases[tone_index, 1],
                    frequency=upper_frequency,
                    frames=frames,
                )
            )

            tone = (
                lower_wave + upper_wave
            ) * 0.5

            output += tone * chord_weights[tone_index]

        output /= sum(chord_weights)

        noise = (
            self.random.standard_normal(frames)
            * self.noise_amount
        )

        output += noise

        # Gentle saturation keeps peaks controlled and softens the sound.
        output = np.tanh(output * 1.25)

        return output.astype(np.float32)

    def _get_chord_frequencies(
        self,
    ) -> list[float]:
        intervals = self.CHORD_INTERVALS[
            self.chord_type
        ]

        return [
            self.frequency * (2.0 ** (semitones / 12.0))
            for semitones in intervals
        ]

    def _generate_soft_wave(
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

        saw = 2.0 * (
            wrapped - 0.5
        )

        sine = np.sin(
            2.0 * np.pi * wrapped
        )

        # Mostly saw for intelligibility, softened with sine.
        wave = (
            0.8 * saw
            + 0.2 * sine
        )

        new_phase = (
            phases[-1] + increment
        ) % 1.0

        return wave, new_phase