from __future__ import annotations


class AnimatedValue:
    """
    Smoothly interpolates between values.

    Example:
        value = AnimatedValue(0.0)

        value.set_target(1.0)

        while True:
            value.update()

            print(value.value)
    """

    def __init__(
        self,
        initial: float = 0.0,
        speed: float = 0.18,
    ) -> None:
        self.value = initial
        self.target = initial
        self.speed = speed

    def set_target(self, target: float) -> None:
        self.target = target

    def update(self) -> float:
        self.value += (
            self.target - self.value
        ) * self.speed

        return self.value

    def snap(self, value: float) -> None:
        """
        Instantly move to a value.
        """
        self.value = value
        self.target = value

    def is_animating(self) -> bool:
        return abs(self.target - self.value) > 1e-4