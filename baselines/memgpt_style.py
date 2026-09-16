from typing import Dict, List


class MemGPTStyleBaseline:
    """B5: Paged memory agent architecture split into working and archival stores."""

    def __init__(self, max_working: int = 5):
        self.working_memory = []
        self.archival_memory = []
        self.max_working = max_working

    def add_memory(self, item: Dict):
        if len(self.working_memory) >= self.max_working:
            evicted = self.working_memory.pop(0)
            self.archival_memory.append(evicted)
        self.working_memory.append(item)
