from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from audio.carrier import CarrierSynth
from audio.reverb import StereoReverb
from audio.vocoder import Vocoder
from config.settings import (
    AUDIO_INPUT_DEVICE,
    AUDIO_OUTPUT_DEVICE,
    BLOCK_SIZE,
    MASTER_OUTPUT_LEVEL,
    MIC_INPUT_GAIN,
    PINCH_CALIBRATION_MAX,
    PINCH_CALIBRATION_MIN,
    REVERB_MIX,
    REVERB_ROOM_SIZE,
    REVERB_STEREO_WIDTH,
    SAMPLE_RATE,
    VOCODER_ATTACK_SECONDS,
    VOCODER_BANDS,
    VOCODER_CARRIER_FREQUENCY,
    VOCODER_HARMONY_GAINS,
    VOCODER_HARMONY_INTERVALS,
    VOCODER_HARMONY_LEVEL,
    VOCODER_HARMONY_PANS,
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
    Real-time gesture-controlled harmonized vocoder.

    Each harmony voice has:
    - its own carrier
    - its own vocoder state
    - its own stereo pan position

    Gesture mapping:
    - fingers apart: natural voice
    - fingers together: harmonized vocoder
    """

    def __init__(self) -> None:
        voice_count = len(VOCODER_HARMONY_INTERVALS)

        if voice_count != len(VOCODER_HARMONY_GAINS):
            raise ValueError(
                "VOCODER_HARMONY_INTERVALS and "
                "VOCODER_HARMONY_GAINS must have equal lengths."
            )

        if voice_count != len(VOCODER_HARMONY_PANS):
            raise ValueError(
                "VOCODER_HARMONY_INTERVALS and "
                "VOCODER_HARMONY_PANS must have equal lengths."
            )

        if PINCH_CALIBRATION_MAX <= PINCH_CALIBRATION_MIN:
            raise ValueError(
                "PINCH_CALIBRATION_MAX must be greater than "
                "PINCH_CALIBRATION_MIN."
            )

        self.lock = threading.Lock()

        self.vocoder_makeup_gain = 1.0
        self.makeup_smoothing = 0.08
        self.maximum_makeup_gain = 8.0

        self.wet_mix = self._clamp(
            VOCODER_WET_MIX,
            VOCODER_MIN_WET_MIX,
            VOCODER_MAX_WET_MIX,
        )

        self.root_frequency = VOCODER_CARRIER_FREQUENCY

        self.voice_frequencies = [
            self._transpose_frequency(
                self.root_frequency,
                semitones,
            )
            for semitones in VOCODER_HARMONY_INTERVALS
        ]

        self.voice_gains = np.asarray(
            VOCODER_HARMONY_GAINS,
            dtype=np.float32,
        )

        self.voice_pans = np.asarray(
            VOCODER_HARMONY_PANS,
            dtype=np.float32,
        )

        if not np.any(self.voice_gains > 0.0):
            raise ValueError(
                "At least one harmony gain must be positive."
            )

        if np.any(self.voice_pans < -1.0) or np.any(
            self.voice_pans > 1.0
        ):
            raise ValueError(
                "Harmony pans must be between -1.0 and 1.0."
            )

        self.carriers = [
            CarrierSynth(
                sample_rate=SAMPLE_RATE,
                frequency=frequency,
            )
            for frequency in self.voice_frequencies
        ]

        self.vocoders = [
            Vocoder(
                sample_rate=SAMPLE_RATE,
                num_bands=VOCODER_BANDS,
                low_frequency=VOCODER_LOW_FREQUENCY,
                high_frequency=VOCODER_HIGH_FREQUENCY,
                attack_seconds=VOCODER_ATTACK_SECONDS,
                release_seconds=VOCODER_RELEASE_SECONDS,
                output_gain=VOCODER_OUTPUT_GAIN,
            )
            for _ in self.voice_frequencies
        ]

        self.reverb = StereoReverb(
            sample_rate=SAMPLE_RATE,
            mix=REVERB_MIX,
            room_size=REVERB_ROOM_SIZE,
            stereo_width=REVERB_STEREO_WIDTH,
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

        for vocoder in self.vocoders:
            vocoder.reset()

    def update(
        self,
        gesture_state: GestureState,
    ) -> None:
        hand = gesture_state.primary_hand

        if hand is None:
            return

        normalized_pinch = self._normalize_pinch(
            hand.pinch
        )

        closeness = 1.0 - normalized_pinch

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

    def get_parameters(
        self,
    ) -> tuple[float, float]:
        with self.lock:
            return (
                self.wet_mix,
                self.root_frequency,
            )

    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        try:
            if status:
                print(
                    f"Audio status: {status}",
                    flush=True,
                )

            voice = indata[:, 0].copy()
            voice *= MIC_INPUT_GAIN

            harmony_output = np.zeros(
                (frames, 2),
                dtype=np.float32,
            )

            for carrier, vocoder, gain, pan in zip(
                self.carriers,
                self.vocoders,
                self.voice_gains,
                self.voice_pans,
            ):
                carrier_block = carrier.generate(
                    frames
                )

                vocoded_voice = vocoder.process(
                    modulator=voice,
                    carrier=carrier_block,
                )

                left_gain, right_gain = (
                    self._equal_power_pan(
                        float(pan)
                    )
                )

                harmony_output[:, 0] += (
                    vocoded_voice
                    * gain
                    * left_gain
                )

                harmony_output[:, 1] += (
                    vocoded_voice
                    * gain
                    * right_gain
                )

            harmony_output *= VOCODER_HARMONY_LEVEL

            with self.lock:
                wet_mix = self.wet_mix

            epsilon = 1e-6

            dry_rms = float(
                np.sqrt(
                    np.mean(
                        voice * voice
                    )
                    + epsilon
                )
            )

            wet_rms = float(
                np.sqrt(
                    np.mean(
                        harmony_output
                        * harmony_output
                    )
                    + epsilon
                )
            )

            if (
                dry_rms > 0.001
                and wet_rms > 0.0001
            ):
                target_makeup_gain = (
                    dry_rms / wet_rms
                )

                target_makeup_gain = self._clamp(
                    target_makeup_gain,
                    0.25,
                    self.maximum_makeup_gain,
                )

                self.vocoder_makeup_gain += (
                    target_makeup_gain
                    - self.vocoder_makeup_gain
                ) * self.makeup_smoothing

            matched_harmony = (
                harmony_output
                * self.vocoder_makeup_gain
            )

            dry_voice = np.column_stack(
                (
                    voice,
                    voice,
                )
            ).astype(np.float32)

            dry_gain = np.cos(
                wet_mix * np.pi / 2.0
            )

            wet_gain = np.sin(
                wet_mix * np.pi / 2.0
            )

            output = (
                dry_gain * dry_voice
                + wet_gain * matched_harmony
            )

            output = self.reverb.process(
                output
            )

            output *= MASTER_OUTPUT_LEVEL

            output = np.tanh(
                output
            ).astype(np.float32)

            outdata[:] = output

        except Exception as error:
            # Audio callback exceptions can otherwise result in unexplained silence.
            outdata.fill(0)

            print(
                f"Audio callback error: {error}",
                flush=True,
            )

    def _normalize_pinch(
        self,
        pinch: float,
    ) -> float:
        calibrated = (
            pinch - PINCH_CALIBRATION_MIN
        ) / (
            PINCH_CALIBRATION_MAX
            - PINCH_CALIBRATION_MIN
        )

        return self._clamp(
            calibrated,
            0.0,
            1.0,
        )

    @staticmethod
    def _equal_power_pan(
        pan: float,
    ) -> tuple[float, float]:
        """
        Convert a -1.0 to 1.0 pan value into left and right gains.
        """
        pan = max(
            -1.0,
            min(1.0, pan),
        )

        angle = (
            pan + 1.0
        ) * np.pi / 4.0

        left_gain = float(
            np.cos(angle)
        )

        right_gain = float(
            np.sin(angle)
        )

        return (
            left_gain,
            right_gain,
        )

    @staticmethod
    def _transpose_frequency(
        frequency: float,
        semitones: float,
    ) -> float:
        return frequency * (
            2.0 ** (semitones / 12.0)
        )

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