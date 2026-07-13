import numpy as np


class Oscillator:
    def __init__(
        self,
        sample_rate: int = 44_100,
        waveform: str = "sine",
    ) -> None:
        self.sample_rate = sample_rate
        self.waveform = waveform
        self.phase = 0.0

    def generate(
        self,
        frequency: float,
        frames: int,
    ) -> np.ndarray:
        phase_increment = (
            2.0
            * np.pi
            * frequency
            / self.sample_rate
        )

        phases = (
            self.phase
            + phase_increment * np.arange(frames)
        )

        samples = self._waveform_samples(phases)

        self.phase = (
            phases[-1] + phase_increment
        ) % (2.0 * np.pi)

        return samples.astype(np.float32)

    def set_waveform(self, waveform: str) -> None:
        allowed_waveforms = {
            "sine",
            "square",
            "sawtooth",
        }

        if waveform not in allowed_waveforms:
            raise ValueError(
                f"Unsupported waveform: {waveform}"
            )

        self.waveform = waveform

    def _waveform_samples(
        self,
        phases: np.ndarray,
    ) -> np.ndarray:
        if self.waveform == "square":
            return np.sign(np.sin(phases))

        if self.waveform == "sawtooth":
            normalized_phase = phases / (2.0 * np.pi)
            return 2.0 * (
                normalized_phase
                - np.floor(normalized_phase + 0.5)
            )

        return np.sin(phases)