from __future__ import annotations

import time

import bootstrap
import numpy as np
import sounddevice as sd

from audio.carrier import CarrierSynth
from audio.vocoder import Vocoder
from config.settings import (
    AUDIO_INPUT_DEVICE,
    AUDIO_OUTPUT_DEVICE,
    BLOCK_SIZE,
    SAMPLE_RATE,
    VOCODER_ATTACK_SECONDS,
    VOCODER_BANDS,
    VOCODER_CARRIER_FREQUENCY,
    VOCODER_CHORD_TYPE,
    VOCODER_HIGH_FREQUENCY,
    VOCODER_LOW_FREQUENCY,
    VOCODER_OUTPUT_GAIN,
    VOCODER_RELEASE_SECONDS,
    VOCODER_WET_MIX,
)


INPUT_GAIN = 2.0
OUTPUT_LEVEL = 0.5


def main() -> None:
    vocoder = Vocoder(
        sample_rate=SAMPLE_RATE,
        num_bands=VOCODER_BANDS,
        low_frequency=VOCODER_LOW_FREQUENCY,
        high_frequency=VOCODER_HIGH_FREQUENCY,
        attack_seconds=VOCODER_ATTACK_SECONDS,
        release_seconds=VOCODER_RELEASE_SECONDS,
        output_gain=VOCODER_OUTPUT_GAIN,
    )

    carrier = CarrierSynth(
        sample_rate=SAMPLE_RATE,
        frequency=VOCODER_CARRIER_FREQUENCY,
        chord_type=VOCODER_CHORD_TYPE,
    )

    wet_mix = max(
        0.0,
        min(1.0, VOCODER_WET_MIX),
    )

    def audio_callback(
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
        voice *= INPUT_GAIN

        carrier_block = carrier.generate(
            frames
        )

        vocoded = vocoder.process(
            modulator=voice,
            carrier=carrier_block,
        )

        output = (
            (1.0 - wet_mix) * voice
            + wet_mix * vocoded
        )

        output *= OUTPUT_LEVEL

        output = np.clip(
            output,
            -1.0,
            1.0,
        )

        if outdata.shape[1] == 1:
            outdata[:, 0] = output
        else:
            outdata[:] = output[:, None]

    print("Starting the vocoder test.")
    print()
    print("Speak or sing into the microphone.")
    print("The carrier uses three detuned saw oscillators.")
    print("Press Ctrl+C to stop.")
    print()
    print(f"Vocoder bands: {VOCODER_BANDS}")
    print(f"Wet mix: {wet_mix:.2f}")
    print(
        "Carrier frequency: "
        f"{VOCODER_CARRIER_FREQUENCY:.1f} Hz"
    )
    print(f"Chord type: {VOCODER_CHORD_TYPE}")
    print()

    stream = sd.Stream(
        device=(
            AUDIO_INPUT_DEVICE,
            AUDIO_OUTPUT_DEVICE,
        ),
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        channels=(1, 2),
        latency="low",
        callback=audio_callback,
    )

    try:
        with stream:
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping vocoder.")

    finally:
        vocoder.reset()


if __name__ == "__main__":
    main()