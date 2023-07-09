from metrics.tod_metrics_base import TodMetricsBase
from my_enums import SpecialPredictions, SpecialTokens
import numpy as np


class IntentAccuracyMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("pred_intents", [], dist_reduce_fx="cat")
        self.add_state("target_intents", [], dist_reduce_fx="cat")

    def _update(self, predictions: list[str], references: list[str]) -> None:
        for target, prediction in zip(references, predictions):
            t_intents = self._extract_section_from_text(
                target,
                SpecialTokens.begin_intent,
                SpecialTokens.end_intent,
                multiple_values=True,
                default_value=[],
            )
            if not len(t_intents):
                continue
            for _ in t_intents:
                self.target_intents.append(1)
            p = self._extract_section_from_text(
                prediction,
                SpecialTokens.begin_intent,
                SpecialTokens.end_intent,
                [SpecialPredictions.DUMMY],
                multiple_values=True,
                trim_spaces=True,
            )
            for t_intent in t_intents:
                if t_intent in p:
                    self.pred_intents.append(1)
                else:
                    self.pred_intents.append(0)

    def _compute(self) -> float:
        try:
            acc = (
                np.array(self.pred_intents) == np.array(self.target_intents)
            ).sum() / len(self.target_intents)
        except ZeroDivisionError:
            acc = 0
        return acc

    def __str__(self) -> str:
        score = self.compute()
        return f"Intent Accuracy:{score*100:.2f}"
