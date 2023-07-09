import numpy as np
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import ResponseMetricType, SpecialTokens
import evaluate
import uuid


class ResponseMetric(TodMetricsBase):
    def __init__(self, metric_name="bleu", metric_key_name=None) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.metric = (
            evaluate.load("rouge", experiment_id=str(uuid.uuid4()))
            if metric_name == ResponseMetricType.ROUGE
            else evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        )
        self.metric_key_name = metric_key_name or metric_name
        self.add_state("pred_responses", [], dist_reduce_fx="cat")
        self.add_state("target_responses", [], dist_reduce_fx="cat")

    def _update(self, predictions: list[str], references: list[str]) -> None:
        pred_responses_batch = []
        target_responses_batch = []
        for pred, ref in zip(predictions, references):
            target_response = self._extract_section_from_text(
                ref,
                SpecialTokens.begin_response,
                SpecialTokens.end_response,
            )
            if not target_response:
                continue
            pred_response = self._extract_section_from_text(
                pred,
                SpecialTokens.begin_response,
                SpecialTokens.end_response,
                "",
            )

            pred_responses_batch.append(pred_response)
            if self.metric_name == ResponseMetricType.ROUGE:
                target_responses_batch.append(target_response)
            elif self.metric_name == ResponseMetricType.BLEU:
                target_responses_batch.append([target_response])
        if len(target_responses_batch) == 0:
            return
        self.metric.add_batch(
            predictions=pred_responses_batch, references=target_responses_batch
        )

    def _compute(self) -> float:
        try:
            out = self.metric.compute()
            res = out[self.metric_key_name]
        except (ZeroDivisionError, ValueError):
            return 0.0

        if self.metric_name == ResponseMetricType.ROUGE:
            return res.mid.fmeasure
        return res

    def __str__(self) -> str:
        score = self.compute()
        return f"Response {self.metric_name.upper()}:{score*100:.2f}"
