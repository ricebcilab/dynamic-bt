"""Base class and shared constants for edge completion criteria."""

from abc import ABC, abstractmethod


AXIS_MAP = {
    "x": [0], "y": [1], "z": [2],
    "xy": [0, 1], "xz": [0, 2], "yz": [1, 2],
    "xyz": [0, 1, 2],
}


class BaseCriteria(ABC):
    """Base class for edge completion criteria."""

    def __init__(self):
        self.message = {}

    def reset(self):
        """Reset stateful data between activations."""
        self.message = {}

    @abstractmethod
    def check(self, *args, **kwargs) -> bool:
        """Return True if the criteria is satisfied.

        On success, populate self.message with relevant info.
        """
