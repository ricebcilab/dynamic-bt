"""Truthiness check with optional negation and duration debounce."""

import time

from .base_criteria import BaseCriteria


class StateCheck(BaseCriteria):
    """Check truthiness of a value, with optional negation and duration debounce."""

    def __init__(self, negate: bool = False, duration: float = 0.0, **kwargs):
        super().__init__()
        self.negate = negate
        self.duration = duration
        self.last_check_ts = None

    def reset(self):
        super().reset()
        self.last_check_ts = None

    def check(self, value) -> bool:
        result = (not bool(value)) if self.negate else bool(value)

        if not result:
            # Condition lapsed — restart the debounce timer next time it goes True
            self.last_check_ts = None
            return False

        if self.last_check_ts is None:
            self.last_check_ts = time.monotonic()

        return (time.monotonic() - self.last_check_ts) >= self.duration
