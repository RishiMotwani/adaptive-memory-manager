from typing import Dict, List


class VanillaRAGBaseline:
    """B3: Unscored, un-decayed static vector database retrieval."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def retrieve(self, query: str, facts: List[Dict]) -> List[Dict]:
        return facts[:self.top_k]
