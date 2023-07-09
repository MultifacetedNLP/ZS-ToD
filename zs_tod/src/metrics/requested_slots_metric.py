import numpy as np
from sgd_dstc8_data_model.dstc_dataclasses import DstcRequestedSlot
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import SpecialTokens
from collections import Counter


class RequestedSlotsMetric(TodMetricsBase):
    def __init_old__(self) -> None:
        super().__init__()
        self.add_state("all_refs", [], dist_reduce_fx="cat")
        self.add_state("all_preds", [], dist_reduce_fx="cat")

    def __init__(self) -> None:
        super().__init__()
        self.add_state("all_f1", [], dist_reduce_fx="cat")

    def __str__(self) -> str:
        score = self.compute()
        return f"Requested Slots F1:{score:.2f}"

    def _update(self, predictions: list[str], references: list[str]) -> any:
        for ref, pred in zip(references, predictions):
            target_txt_items = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )
            target_slots = [
                DstcRequestedSlot.from_string(t).slot_name for t in target_txt_items
            ]

            pred_txt_items = self._extract_section_and_split_items_from_text(
                pred,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )
            pred_slots = [
                DstcRequestedSlot.from_string(t).slot_name for t in pred_txt_items
            ]
            # the code below is from the SGD codebase
            ref_counter = Counter(target_slots)
            pred_counter = Counter(pred_slots)
            true = sum(ref_counter.values())
            positive = sum(pred_counter.values())
            true_positive = sum((ref_counter & pred_counter).values())
            precision = float(true_positive) / positive if positive else 1.0
            recall = float(true_positive) / true if true else 1.0
            if precision + recall > 0.0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:  # The F1-score is defined to be 0 if both precision and recall are 0.
                f1 = 0.0
            self.all_f1.append(f1)

    def _compute(self) -> float:
        return np.mean(self.all_f1) * 100
