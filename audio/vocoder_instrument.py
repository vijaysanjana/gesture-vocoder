from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from audio.carrier import CarrierSynth
from audio.vocoder import Vocoder
from config.settings import (
    AUDIO_INPUT_DEVICE,
    AUDIO_OUTPUT_DEVICE,
    BLOCK_SIZE,
    MASTER_OUTPUT_LEVEL,
    MIC_INPUT_GAIN,
    SAMPLE_RATE,
    VOCODER_ATTACK_SECONDS,
    VOCODER_BANDS,
    VOCODER_CARRIER_FREQUENCY,
    VOCODER_HIGH_FREQUENCY,
    VOCODER_LOW_FREQUENCY,
    VOCODER_MAX_WET_MIX,
    VOCODER_MIN_WET_MIX,
    VOCODER_OUTPUT_GAIN,
    VOCODER_RELEASE_SECONDS,
    VOCODER_WET_MIX,
)
from gestures.gesture_state import GestureState


class VocoderInstrument:
    """
    Real-time gesture-controlled vocoder.

    Pinch mapping:
        fingers together -> more robotic
        fingers apart    -> slightly more natural
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()

        self.wet_mix = self._clamp(
            VOCODER_WET_MIX,
            VOCODER_MIN_WET_MIX,
            VOCODER_MAX_WET_MIX,
        )

        self.carrier_frequency = VOCODER_CARRIER_FREQUENCY

        self.vocoder = Vocoder(
            sample_rate=SAMPLE_RATE,
            num_bands=VOCODER_BANDS,
            low_frequency=VOCODER_LOW_FREQUENCY,
            high_frequency=VOCODER_HIGH_FREQUENCY,
            attack_seconds=VOCODER_ATTACK_SECONDS,
            release_seconds=VOCODER_RELEASE_SECONDS,
            output_gain=VOCODER_OUTPUT_GAIN,
        )

        self.carrier = CarrierSynth(
            sample_rate=SAMPLE_RATE,
            frequency=self.carrier_frequency,
        )

        self.stream = sd.Stream(
            device=(
                AUDIO_INPUT_DEVICE,
                AUDIO_OUTPUT_DEVICE,
            ),
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            channels=(1, 2),
            latency="low",
            callback=self._audio_callback,
        )

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        if self.stream.active:
            self.stream.stop()

        self.stream.close()
        self.vocoder.reset()

    def update(
        self,
        gesture_state: GestureState,
    ) -> None:
        hand = gesture_state.primary_hand

        # Keep the last mix value when the hand briefly disappears.
        if hand is None:
            return

        # Current pinch scale:
        # 0.0 = fingers together
        # 1.0 = fingers apart
        closeness = 1.0 - hand.pinch

        wet_mix = (
            VOCODER_MIN_WET_MIX
            + closeness
            * (
                VOCODER_MAX_WET_MIX
                - VOCODER_MIN_WET_MIX
            )
        )

        with self.lock:
            self.wet_mix = self._clamp(
                wet_mix,
                VOCODER_MIN_WET_MIX,
                VOCODER_MAX_WET_MIX,
            )

    def get_parameters(self) -> tuple[float, float]:
        with self.lock:
            return (
                self.wet_mix,
                self.carrier_frequency,
            )

    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(
                f"Audio status: {status}",
                flush=True,
            )

        voice = indata[:, 0].copy()
        voice *= MIC_INPUT_GAIN

        carrier_block = self.carrier.generate(
            frames
        )

        vocoded = self.vocoder.process(
            modulator=voice,
            carrier=carrier_block,
        )

        with self.lock:
            wet_mix = self.wet_mix

        output = (
            (1.0 - wet_mix) * voice
            + wet_mix * vocoded
        )

        output *= MASTER_OUTPUT_LEVEL

        output = np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)

        if outdata.shape[1] == 1:
            outdata[:, 0] = output
        else:
            outdata[:] = output[:, None]

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(maximum, value),
        )