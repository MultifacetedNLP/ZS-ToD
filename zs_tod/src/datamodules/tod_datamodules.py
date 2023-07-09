from dotmap import DotMap

import torch
from datamodules.base_datamodule import BaseDataModule
from configs.dm_config import DataModuleConfig

from my_enums import (
    Steps,
)
from tod.tod_dataclasses import TodTestDataBatch

from tod.turns.zs_tod_turn import TodTurnCsvRow


class TodDataModule(BaseDataModule):
    _huggingface_ignore_label_id = -100

    def __init__(
        self,
        cfg: DataModuleConfig,
        steps: list[Steps] = None,
        tod_turn_row_cls=TodTurnCsvRow,
    ):
        super().__init__(cfg, steps, tod_turn_row_cls=tod_turn_row_cls)

    def training_collator(self, batch: list[TodTurnCsvRow], is_pretrain: bool = False):
        input_ids = []
        attention_masks = []
        labels = []
        targets_text = []
        for item in batch:
            input_tokens, label, attention_mask = self.collate_single_item(
                item.context,
                item.schema,
                item.target,
                self.cfg.max_token_len,
                is_pretrain,
            )
            input_ids.append(input_tokens)
            attention_masks.append(attention_mask)
            labels.append(label)
            targets_text.append(item.target)

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }

        return out

    def my_test_collate(self, batch: list[TodTurnCsvRow]) -> TodTestDataBatch:
        data = DotMap(
            {
                key: []
                for key in [
                    "input_ids",
                    "attention_masks",
                    "dialog_ids",
                    "turn_ids",
                    "contexts",
                    "schemas",
                    "targets",
                ]
            }
        )
        for item in batch:
            data.dialog_ids.append(item.dialog_id)
            data.turn_ids.append(item.turn_id)
            data.contexts.append(item.context)
            data.targets.append(item.target)
            data.schemas.append(item.schema)

            input_tokens, _, attention_mask = self.collate_single_item(
                "".join(
                    [
                        item.context,
                    ]
                ),
                item.schema,
                "",
                self.cfg.test_prompt_max_len,
                True,
            )
            data.input_ids.append(input_tokens)
            data.attention_masks.append(attention_mask)

        return TodTestDataBatch(
            input_ids=torch.stack(data.input_ids),
            attention_masks=torch.stack(data.attention_masks),
            schemas_text=data.schemas,
            contexts_text=data.contexts,
            targets_text=data.targets,
            dialog_ids=data.dialog_ids,
            turn_ids=data.turn_ids,
        )
