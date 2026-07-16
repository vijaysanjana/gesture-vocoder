from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


class Vocoder:
    """
    A basic real-time channel vocoder.

    The modulator is the microphone/voice signal.
    The carrier is a harmonically rich synth signal.

    Each signal is divided into matching frequency bands. The amplitude
    envelope of each voice band controls the corresponding carrier band.
    """

    def __init__(
        self,
        sample_rate: int,
        num_bands: int = 12,
        low_frequency: float = 100.0,
        high_frequency: float = 8_000.0,
        attack_seconds: float = 0.008,
        release_seconds: float = 0.080,
        output_gain: float = 1.8,
    ) -> None:
        if num_bands < 2:
            raise ValueError("A vocoder requires at least two bands.")

        nyquist = sample_rate / 2.0

        if low_frequency <= 0:
            raise ValueError("low_frequency must be greater than zero.")

        if high_frequency >= nyquist:
            raise ValueError(
                "high_frequency must be below the Nyquist frequency."
            )

        if low_frequency >= high_frequency:
            raise ValueError(
                "low_frequency must be lower than high_frequency."
            )

        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.output_gain = output_gain

        self.attack_coefficient = self._time_to_coefficient(
            attack_seconds
        )
        self.release_coefficient = self._time_to_coefficient(
            release_seconds
        )

        self.band_edges = np.geomspace(
            low_frequency,
            high_frequency,
            num_bands + 1,
        )

        self.filters: list[np.ndarray] = []

        self.modulator_states: list[np.ndarray] = []
        self.carrier_states: list[np.ndarray] = []

        self.envelopes = np.zeros(
            num_bands,
            dtype=np.float64,
        )

        for band_index in range(num_bands):
            low = float(self.band_edges[band_index])
            high = float(self.band_edges[band_index + 1])

            sos = butter(
                N=2,
                Wn=(low, high),
                btype="bandpass",
                fs=sample_rate,
                output="sos",
            )

            self.filters.append(sos)

            state_shape = (
                sos.shape[0],
                2,
            )

            self.modulator_states.append(
                np.zeros(
                    state_shape,
                    dtype=np.float64,
                )
            )

            self.carrier_states.append(
                np.zeros(
                    state_shape,
                    dtype=np.float64,
                )
            )

    def process(
        self,
        modulator: np.ndarray,
        carrier: np.ndarray,
    ) -> np.ndarray:
        """
        Process one mono audio block.

        Both arrays must have the same one-dimensional shape.
        """
        modulator = np.asarray(
            modulator,
            dtype=np.float64,
        ).reshape(-1)

        carrier = np.asarray(
            carrier,
            dtype=np.float64,
        ).reshape(-1)

        if modulator.shape != carrier.shape:
            raise ValueError(
                "Modulator and carrier blocks must have equal shapes."
            )

        output = np.zeros_like(
            modulator,
            dtype=np.float64,
        )

        for band_index, sos in enumerate(self.filters):
            modulator_band, new_modulator_state = sosfilt(
                sos,
                modulator,
                zi=self.modulator_states[band_index],
            )

            carrier_band, new_carrier_state = sosfilt(
                sos,
                carrier,
                zi=self.carrier_states[band_index],
            )

            self.modulator_states[
                band_index
            ] = new_modulator_state

            self.carrier_states[
                band_index
            ] = new_carrier_state

            envelope = self._follow_envelope(
                signal=np.abs(modulator_band),
                initial_value=self.envelopes[band_index],
            )

            self.envelopes[band_index] = envelope[-1]

            output += carrier_band * envelope

        # Compensate somewhat for spreading energy across multiple bands.
        output *= self.output_gain / np.sqrt(self.num_bands)

        return np.tanh(output).astype(np.float32)

    def reset(self) -> None:
        """
        Clear all filter and envelope state.
        """
        self.envelopes.fill(0.0)

        for state in self.modulator_states:
            state.fill(0.0)

        for state in self.carrier_states:
            state.fill(0.0)

    def _follow_envelope(
        self,
        signal: np.ndarray,
        initial_value: float,
    ) -> np.ndarray:
        envelope = np.empty_like(
            signal,
            dtype=np.float64,
        )

        previous = initial_value

        for index, sample in enumerate(signal):
            coefficient = (
                self.attack_coefficient
                if sample > previous
                else self.release_coefficient
            )

            previous = (
                coefficient * previous
                + (1.0 - coefficient) * sample
            )

            envelope[index] = previous

        return envelope

    def _time_to_coefficient(
        self,
        seconds: float,
    ) -> float:
        if seconds <= 0:
            return 0.0

        return float(
            np.exp(
                -1.0
                / (
                    seconds
                    * self.sample_rate
                )
            )
        )