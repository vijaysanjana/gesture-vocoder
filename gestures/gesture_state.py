from dataclasses import dataclass, field
from typing import Any


@dataclass
class HandState:
    """
    A clean representation of one detected hand.

    Values such as pinch and height are normalized between 0 and 1.
    """

    pinch: float
    height: float
    handedness: str
    landmarks: list[Any] = field(default_factory=list)


@dataclass
class GestureState:
    """
    The full gesture state for one video frame.
    """

    left: HandState | None = None
    right: HandState | None = None

    @property
    def has_hand(self) -> bool:
        return self.left is not None or self.right is not None

    @property
    def primary_hand(self) -> HandState | None:
        """
        Use the right hand when available, otherwise use the left hand.
        """
        return self.right or self.left