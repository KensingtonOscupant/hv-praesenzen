import weave
from weave import Scorer
from typing import Any

class AGMPresenceScorer(Scorer):

    @weave.op()
    def score(
        self,
        output: dict | None,
        label_value: str
    ) -> Any:

        pred_value   = output.get("label_value_predicted")

        return {
            "label_predicted_em": pred_value == label_value,
            # not nice because redundant in final overview, but has to be passed through to summarize function
            "label_value_predicted": output.get("label_value_predicted")

        }

    @weave.op()
    def summarize(self, score_rows: list) -> dict | None:
        if not score_rows:
            return None

        correct = sum(r.get("label_predicted_em", False) for r in score_rows)
        abstain = sum(1 for r in score_rows if r.get("label_value_predicted") == -1)
        total = len(score_rows)
        wrong = total - correct - abstain
        accuracy = correct / total if total else 0.0
        selective_accuracy = correct / (correct + wrong)
        coverage = (correct + wrong) / total

        return {
                "total": total,
                "correct": correct,
                "wrong": wrong,
                "abstain": abstain,
                "accuracy": accuracy,
                "selective_accuracy": selective_accuracy,
                "coverage": coverage,
        }
