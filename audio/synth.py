import threading

import numpy as np
import sounddevice as sd

from audio.oscillator import Oscillator
from gestures.gesture_state import GestureState


SAMPLE_RATE = 44_100
MIN_FREQUENCY = 130.81
MAX_FREQUENCY = 1046.50
MAX_VOLUME = 0.15


class Synth:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        waveform: str = "sine",
    ) -> None:
        self.sample_rate = sample_rate

        self.frequency = 440.0
        self.volume = 0.0

        self.lock = threading.Lock()

        self.oscillator = Oscillator(
            sample_rate=sample_rate,
            waveform=waveform,
        )

        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=256,
            callback=self._audio_callback,
        )

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        if self.stream.active:
            self.stream.stop()

        self.stream.close()

    def update(
        self,
        gesture_state: GestureState,
    ) -> None:
        hand = gesture_state.primary_hand

        if hand is None:
            self.set_parameters(
                frequency=440.0,
                volume=0.0,
            )
            return

        frequency = self._height_to_frequency(
            hand.height
        )

        volume = hand.pinch * MAX_VOLUME

        self.set_parameters(
            frequency=frequency,
            volume=volume,
        )

    def set_parameters(
        self,
        frequency: float,
        volume: float,
    ) -> None:
        with self.lock:
            self.frequency = frequency
            self.volume = volume

    def get_parameters(
        self,
    ) -> tuple[float, float]:
        with self.lock:
            return self.frequency, self.volume

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            print(f"Audio status: {status}")

        with self.lock:
            frequency = self.frequency
            volume = self.volume

        samples = self.oscillator.generate(
            frequency=frequency,
            frames=frames,
        )

        outdata[:, 0] = samples * volume

    @staticmethod
    def _height_to_frequency(
        height: float,
    ) -> float:
        height = max(0.0, min(1.0, height))

        return MIN_FREQUENCY * (
            MAX_FREQUENCY / MIN_FREQUENCY
        ) ** height