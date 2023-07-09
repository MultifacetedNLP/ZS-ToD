from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union
from omegaconf import ListConfig

import torch
from torch.utils.data import DataLoader, Dataset
from configs.dataprep_config import DataPrepConfig
from configs.dm_config import DataModuleConfig

from tod.turns.zs_tod_turn import TodTurnCsvRow
import utils
from my_enums import SpecialTokens, Steps
from zs_tod_dstc_data_prep import ZsTodDSTCDataPrep
import copy
import random

random.seed(420)


@dataclass(frozen=True)
class StepData:
    name: Steps
    num_dialog: int
    overwrite: bool
    split_percent: float
    domain_settings: Union[list[str], str]


class BaseDataModule(ABC):
    _huggingface_ignore_label_id = -100
    domain_step_map = {
        Steps.TRAIN: f"{Steps.TRAIN.value}_domain_settings",
        Steps.DEV: f"{Steps.DEV.value}_domain_settings",
        Steps.TEST: f"{Steps.TEST.value}_domain_settings",
    }

    def __init__(
        self,
        cfg: DataModuleConfig,
        steps: list[Steps],
        tod_turn_row_cls=TodTurnCsvRow,
    ):
        self.cfg = cfg
        self.tod_turn_row_cls = tod_turn_row_cls
        self.datasets: dict[str, TodDataSet] = {}
        self.grouped_test_datasets: list[str, TodDataSet] = {}
        self.steps = steps
        self.setup()
        self.prompt_token_map = {}

    @abstractmethod
    def training_collator(
        self,
        batch: list[TodTurnCsvRow],
        is_pretrain=False,
    ):
        return ValueError("Not implemented")

    @abstractmethod
    def my_test_collate(self, batch: list[TodTurnCsvRow]):
        return ValueError("Not implemented")

    def prepare_data(self, stdp: ZsTodDSTCDataPrep):
        stdp.run()

    def setup_single_run(
        self, step: str, step_data: StepData, domain_setting: Union[str, list[str]]
    ) -> "TodDataSet":
        cfg = copy.deepcopy(self.cfg)
        cfg.step_name = step_data.name
        cfg.num_dialogs = step_data.num_dialog
        cfg.overwrite = step_data.overwrite
        cfg.domain_setting = domain_setting
        stdp = ZsTodDSTCDataPrep(DataPrepConfig.from_dm_config(cfg))
        self.prepare_data(stdp)
        csv_path = utils.get_csv_data_path(
            step,
            step_data.num_dialog,
            cfg=stdp.cfg,
        )
        try:
            data = utils.read_csv_dataclass(csv_path, self.tod_turn_row_cls)
        except FileNotFoundError:
            data = []

        data = self.get_data_by_split_percent(data, step_data.split_percent)
        return TodDataSet(data)

    def setup(self):
        for step in self.steps:
            step_data = self.get_step_data(step)
            if isinstance(step_data.domain_settings[0], ListConfig):
                self.datasets[step] = []
                for domain_setting in step_data.domain_settings:
                    self.datasets[step].append(
                        self.setup_single_run(step, step_data, domain_setting)
                    )
            else:
                self.datasets[step] = self.setup_single_run(
                    step,
                    step_data,
                    step_data.domain_settings,
                )

    def get_step_data(self, step: Steps) -> StepData:
        index = Steps.get_index(step)
        return StepData(
            step,
            self.cfg.num_dialogs[index],
            self.cfg.overwrite[index],
            self.cfg.data_split_percent[index],
            getattr(self.cfg, self.domain_step_map[step]),
        )

    def get_data_by_split_percent(
        self, data: list[TodTurnCsvRow], split_percent: float
    ):
        return data[: int(len(data) * split_percent)]

    def _check_if_step_data_exists(
        self,
        step: Steps = Steps.TRAIN,
        msg: str = "There is no train data, so cannot create dev/test",
    ):
        if not self.datasets[step]:
            raise ValueError(msg)

    def test_dataloader(self) -> any:
        dls = self.datasets[Steps.TEST]
        if not isinstance(dls, list):
            dls = [dls]

        return [
            (
                DataLoader(
                    dl,
                    batch_size=self.cfg.test_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    collate_fn=self.my_test_collate,
                    pin_memory=True,
                ),
                domain_setting,
            )
            for dl, domain_setting in zip(dls, self.cfg.test_domain_settings)
        ]

    def _get_token_id(self, text: str) -> int:
        return self.cfg.tokenizer.encode(text)[0]

    def train_tokenizer(self, item):
        try:
            tokens = self.cfg.tokenizer.encode(
                item,
                return_tensors="pt",
            )
        except Exception as e:
            tokens = torch.empty([1, 0], dtype=torch.int32)
        return tokens.to(dtype=torch.int32)

    def get_training_labels(self, context_len, unused_len, target_tokens):
        return torch.cat(
            [
                torch.full([context_len], self._huggingface_ignore_label_id),
                target_tokens,
                torch.full([unused_len], self._huggingface_ignore_label_id),
            ]
        )

    def pretraining_collator(self, batch: list[TodTurnCsvRow]):
        return self.training_collator(batch, True)

    def collate_single_item(
        self,
        context: str,
        schema: str,
        target: str,
        max_length: int,
        dont_create_labels: bool,
    ):
        context_tokens = self.train_tokenizer(context)[0]
        schema_tokens = self.train_tokenizer(schema)[0]
        target_tokens = self.train_tokenizer(target)[0]
        unused_len = (
            max_length - len(context_tokens) - len(schema_tokens) - len(target_tokens)
        )
        if len(schema_tokens) > max_length:
            raise ValueError("Schema is too long")
        if len(target_tokens) > max_length:
            raise ValueError("Target is too long")
        if unused_len < 0:
            context_start_tokens = context_tokens[:1]
            trimmed_context = context_tokens[unused_len * -1 + 1 :]
            context_tokens = torch.cat([context_start_tokens, trimmed_context], axis=0)
            unused_len = 0
        pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
        input_tokens = torch.cat([schema_tokens, context_tokens, target_tokens, pad])
        if dont_create_labels:
            label = input_tokens
        else:
            label = torch.cat(
                [
                    torch.full(
                        [len(context_tokens) + len(schema_tokens)],
                        self._huggingface_ignore_label_id,
                    ),
                    target_tokens,
                    torch.full([unused_len], self._huggingface_ignore_label_id),
                ]
            )
        attention_mask = input_tokens.ne(self.cfg.tokenizer.pad_token_id).to(
            torch.int32
        )
        return input_tokens, label, attention_mask


class TodDataSet(Dataset):
    def __init__(
        self,
        data: List[TodTurnCsvRow] = [],
    ):
        self.data: list[TodTurnCsvRow] = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> TodTurnCsvRow:
        return self.data[idx]
