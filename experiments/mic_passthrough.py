from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

import bootstrap  # noqa: F401
from config.settings import (
    AUDIO_INPUT_DEVICE,
    AUDIO_OUTPUT_DEVICE,
    BLOCK_SIZE,
    SAMPLE_RATE,
)


MONITOR_GAIN = 0.6


def audio_callback(
    indata: np.ndarray,
    outdata: np.ndarray,
    frames: int,
    time_info,
    status: sd.CallbackFlags,
) -> None:
    """
    Copy microphone input directly to headphone output.

    This is intentionally simple. The vocoder will eventually replace
    this direct copy with processed output.
    """
    if status:
        print(f"Audio status: {status}")

    if indata.shape[1] == 1:
        mono_input = indata[:, 0]
    else:
        mono_input = np.mean(
            indata,
            axis=1,
        )

    monitored_signal = mono_input * MONITOR_GAIN

    # Protect against accidental clipping.
    monitored_signal = np.clip(
        monitored_signal,
        -1.0,
        1.0,
    )

    if outdata.shape[1] == 1:
        outdata[:, 0] = monitored_signal
    else:
        outdata[:] = monitored_signal[:, None]


def main() -> None:
    print("Starting microphone passthrough.")
    print("Speak into the microphone.")
    print("Press Ctrl+C to stop.")
    print()
    print(f"Input device: {AUDIO_INPUT_DEVICE}")
    print(f"Output device: {AUDIO_OUTPUT_DEVICE}")

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
        print("\nStopping microphone passthrough.")


if __name__ == "__main__":
    main()