from dataclasses import dataclass
from typing import Union
import torch
import numpy as np
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import GoalMetricConfigType, SpecialTokens
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief


@dataclass
class GoalMetricConfig:
    start_token: str
    end_token: str
    step_name: GoalMetricConfigType
    tod_class: Union[ZsTodBelief, ZsTodAction]


class GoalMetricConfigFactory:
    @classmethod
    def create(self, step) -> GoalMetricConfig:
        if step == GoalMetricConfigType.ACTION:
            return GoalMetricConfig(
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
                GoalMetricConfigType.ACTION,
                ZsTodAction,
            )
        elif step == GoalMetricConfigType.BELIEF:
            return GoalMetricConfig(
                SpecialTokens.begin_belief,
                SpecialTokens.end_belief,
                GoalMetricConfigType.BELIEF,
                ZsTodBelief,
            )
        elif step == GoalMetricConfigType.USER_ACTION:
            return GoalMetricConfig(
                SpecialTokens.begin_user_action,
                SpecialTokens.end_user_action,
                GoalMetricConfigType.USER_ACTION,
                ZsTodAction,
            )
        else:
            raise ValueError(f"Unknown step name: {step}")


class GoalMetric(TodMetricsBase):
    """
    Computes avg and joint goal accuracy of belief or action.
    args:
        target_step_class: One of the following
            * ZsTodAction: it will calculate action accuracy
            * ZsTodBelief: it will calculate belief accuracy
    """

    def __init__(
        self, config: GoalMetricConfig, slot_categories: dict[str, bool]
    ) -> None:
        super().__init__()
        self.slot_categories = slot_categories
        self.all_accuracies = []
        self.joint_accuracies = []
        self.wrong_preds = {}
        self.config = config
        self.add_state("joint_accuracies", [], dist_reduce_fx="cat")
        self.add_state("all_accuracies", [], dist_reduce_fx="cat")

    def _update(self, turn_predictions: list[str], references: list[str]) -> None:
        for ref, pred in zip(references, turn_predictions):
            multiple_values = True if self.config.tod_class == ZsTodBelief else False
            target_txt_items = self._extract_section_and_split_items_from_text(
                ref,
                self.config.start_token,
                self.config.end_token,
                multiple_values=multiple_values,
            )
            if not len(target_txt_items):
                continue
            target_items = [
                self.config.tod_class.from_string(t, self.slot_categories)
                for t in target_txt_items
            ]
            if self.config.tod_class is ZsTodBelief:
                target_items = [t for t in target_items if t.values]
            if not len(target_items):
                continue
            pred_belief_txt_items = self._extract_section_and_split_items_from_text(
                pred,
                self.config.start_token,
                self.config.end_token,
                multiple_values=multiple_values,
                trim_spaces=True,
            )
            pred_beliefs = [
                self.config.tod_class.from_string(t, self.slot_categories)
                for t in pred_belief_txt_items
            ]

            turn_predictions = []
            for t in target_items:
                if t in pred_beliefs:
                    turn_predictions.append(1)
                else:
                    turn_predictions.append(0)
            self.joint_accuracies.append(
                torch.tensor(np.prod(turn_predictions))
                if len(turn_predictions)
                else torch.tensor(0)
            )
            self.all_accuracies.append(
                np.mean(turn_predictions) if len(turn_predictions) else 0
            )

    def _compute(self) -> float:
        return np.mean(self.all_accuracies), np.mean(self.joint_accuracies)

    def __str__(self) -> str:
        avg_ga, joint_ga = self.compute()
        return f"Average {self.config.step_name} Accuracy:{avg_ga*100:.2f}\nJoint {self.config.step_name} Accuracy:{joint_ga*100:.2f}"
