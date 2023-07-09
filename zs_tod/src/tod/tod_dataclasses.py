from dataclasses import dataclass, field
import numpy as np

# Datamodule classes


@dataclass
class TodTestDataBatch:
    input_ids: list[list[int]]
    attention_masks: list[list[int]]
    contexts_text: list[str]
    schemas_text: list[str]
    targets_text: list[str]
    dialog_ids: list[int]
    turn_ids: list[int]


@dataclass
class PredRef:
    pred: str
    ref: str


class InferenceRecords:
    def __init__(self):
        self.preds = []
        self.dialog_ids = []
        self.turn_ids = []
        self.refs = []
        self.contexts = []
        self.is_data_concatenated = False

    def add(self, preds, refs, dialog_ids, turn_ids, contexts):
        self.preds.append(preds)
        self.dialog_ids.append(dialog_ids)
        self.turn_ids.append(turn_ids)
        self.refs.append(refs)
        self.contexts.append(contexts)

    def concat_data(self):
        self.preds = np.concatenate(self.preds, axis=0)
        self.refs = np.concatenate(self.refs, axis=0)
        self.dialog_ids = np.concatenate(self.dialog_ids, axis=0)
        self.turn_ids = np.concatenate(self.turn_ids, axis=0)
        self.contexts = np.concatenate(self.contexts, axis=0)
        self.is_data_concatenated = True
