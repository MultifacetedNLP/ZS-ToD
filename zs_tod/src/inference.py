from typing import Tuple, Union

import numpy as np


from tqdm import tqdm


from metrics.intent_accuracy_metric import IntentAccuracyMetric
from metrics.response_metrics import ResponseMetric


from torchmetrics import MetricCollection
from metrics.goal_metric import GoalMetric, GoalMetricConfigFactory
from metrics.requested_slots_metric import RequestedSlotsMetric
from metrics.dstc_metrics import InformMetric, SuccessMetric, CombinedMetric
from my_enums import GoalMetricConfigType
import utils
from sgd_dstc8_data_model.dstc_dataclasses import get_slot_categories

from zs_tod.src.tod.tod_dataclasses import InferenceRecords


class Inference:
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg
        self._set_metrics()

    def test(self):
        self.cfg.logger.info(self.cfg.out_dir)
        test_dl_func = (
            self.cfg.datamodule.grouped_test_dataloader
            if self.cfg.test_num_turns_groups
            else self.cfg.datamodule.test_dataloader
        )

        for test_dataloader, domain_setting in test_dl_func():
            domains_str = utils.get_domain_setting_str(domain_setting)
            test_csv_out_data = []
            text_csv_out_path = f"simple_tod_dstc_predictions_{domains_str}_{self.cfg.num_turns}_dialogs_{self.cfg.num_test_dialogs}.csv"
            if not len(test_dataloader):
                self.cfg.logger.info(f"No data to test for {domains_str}")
                continue
            inf_records = InferenceRecords()

            for batch in tqdm(test_dataloader):
                pred_text_no_pad = self.cfg.generation_handler.get_generation(
                    batch,
                    self.cfg.max_token_len,
                    self.cfg.test_prompt_max_len,
                    self.cfg.postprocess_generation,
                )
                self.tod_metrics.update(
                    references=batch.targets_text, predictions=pred_text_no_pad
                )
                self.combined_metrics.update(
                    references=batch.targets_text, predictions=pred_text_no_pad
                )
                inf_records.add(
                    pred_text_no_pad,
                    batch.targets_text,
                    batch.dialog_ids,
                    batch.turn_ids,
                    batch.contexts_text,
                )

            inf_records.concat_data()
            test_csv_out_data = np.column_stack(
                [
                    inf_records.dialog_ids,
                    inf_records.turn_ids,
                    inf_records.contexts,
                    inf_records.refs,
                    inf_records.preds,
                ]
            )
            headers = ["dialog_id", "turn_id", "context", "target", "prediction"]
            utils.write_csv(headers, test_csv_out_data, text_csv_out_path)
            self.cfg.logger.info(f"Testing {domains_str}")
            self._print_metrics()

    def _print_metrics(self) -> Tuple[list[str], list[str]]:
        tod_metrics_str = [str(self.tod_metrics[m]) for m in self.tod_metrics]
        combined_metrics_str = [
            str(self.combined_metrics[m]) for m in self.combined_metrics
        ]
        all_metric_str = "\n".join(
            np.concatenate([tod_metrics_str, combined_metrics_str])
        )
        metric_strs = all_metric_str.split("\n")
        cols = []
        header_sep = []
        values = []
        for metric_str in metric_strs:
            if not metric_str:
                continue
            col, value = metric_str.split(":")
            cols.append(col)
            header_sep.append("-")
            values.append(value)
        self.cfg.logger.info(f"|{'|'.join(cols)}|")
        self.cfg.logger.info(f"|{'|'.join(header_sep)}|")
        self.cfg.logger.info(f"|{'|'.join(values)}|")
        return cols, values

    def _set_metrics(self):
        slot_categories = get_slot_categories(self.cfg.raw_data_root)

        tod_metrics = {}
        combined_metrics = {}
        tod_metrics.update(
            {
                "goal_accuracy": GoalMetric(
                    GoalMetricConfigFactory.create(GoalMetricConfigType.BELIEF),
                    slot_categories,
                ),
                "intent_accuracy": IntentAccuracyMetric(),
                "requested_slots": RequestedSlotsMetric(),
            }
        )
        tod_metrics.update(
            {
                "inform": InformMetric(),
                "success": SuccessMetric(slot_categories),
                "action_accuracy": GoalMetric(
                    GoalMetricConfigFactory.create(GoalMetricConfigType.ACTION),
                    slot_categories,
                ),
                "user_action_accuracy": GoalMetric(
                    GoalMetricConfigFactory.create(GoalMetricConfigType.USER_ACTION),
                    slot_categories,
                ),
            }
        )
        tod_metrics.update(
            {
                "response_bleu": ResponseMetric(
                    metric_name="bleu", metric_key_name="google_bleu"
                ),
            }
        )

        combined_metrics.update(
            {
                "combined": CombinedMetric(
                    tod_metrics["inform"],
                    tod_metrics["success"],
                    tod_metrics["response_bleu"],
                ),
            }
        )
        self.tod_metrics = MetricCollection(tod_metrics)
        self.combined_metrics = MetricCollection(combined_metrics)

    def _get_token_id(self, text: str) -> int:
        return self.cfg.tokenizer.encode(text)[0]
