import weave
from weave import Scorer
from typing import Any


class AGMPresenceScorer(Scorer):

    # ---------- per-row --------------------------------------------------
    @weave.op()
    def score(
        self,
        output: dict | None,
        label_value: str
    ) -> Any:

        pred_value   = output.get("label_value_predicted")

        return {
            "label_predicted_em": pred_value == label_value,
        }

    @weave.op()
    def summarize(self, score_rows: list) -> dict | None:
        if not score_rows:
            return None

        # value-step accuracy
        correct_value = sum(r.get("label_predicted_em", False) for r in score_rows)
        total         = len(score_rows)
        accuracy      = correct_value / total if total else 0.0

        return {
            "value": {
                "accuracy": accuracy,
                "correct": correct_value,
                "total": total
            }
        }
