from typing import Dict, List


class SlidingWindowBaseline:
    """B2: Truncates historical context strictly to the last N turns."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def process(self, history: List[Dict]) -> List[Dict]:
        return history[-self.window_size:]
