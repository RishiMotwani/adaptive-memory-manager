from typing import Dict, List


class SummarizationOnlyBaseline:
    """B4: Periodically collapses past context into monolithic text blocks without category scoring."""

    def summarize(self, history: List[Dict]) -> Dict:
        combined = " ".join([h["user"] for h in history])
        return {"fact": f"Global Summary: {combined[:150]}...", "category": "summary"}
