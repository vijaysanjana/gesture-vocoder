import sounddevice as sd


def main() -> None:
    print(sd.query_devices())
    print()
    print("Default input/output devices:")
    print(sd.default.device)


if __name__ == "__main__":
    main()