import copy
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
import humps
from configs.dataprep_config import DataPrepConfig
from my_enums import Steps, ZsTodConstants
from tod.turns.general_turn_csv_row import GeneralTurnCsvRow
from tod.turns.turn_csv_row import TurnCsvRowBase
from tod.turns.zs_tod_turn import ZsTodTurn
from tod.zs_target import ZsTodTarget
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief
from tod.zs_tod_context import ZsTodContext
from tod.zs_tod_dst import ZsTodDst
import utils
from pathos.multiprocessing import ProcessingPool as Pool

from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcDialog,
    DstcFrame,
    DstcSchema,
    DstcTurn,
    get_schemas,
)
from utils import get_csv_data_path, get_dialog_file_paths

from my_enums import SpecialTokens


class ZsTodDSTCDataPrep:
    def __init__(self, cfg: DataPrepConfig):
        self.cfg = cfg

    """
        A context contains a list of user and system turns. The data format expects system turn first, and then user turn.
        
        In the first turn, system turn is null and there is only a user turn and the system turn is placed in 
        the next system utterance of the current context.

        If we have a previous turn, we make a deep copy of it. Check context length by number of turns.
        The system utterance for this turn is the next system utterance of the previous context.
    """

    def _prepare_context(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        prev_tod_turn: ZsTodTurn,
        schemas: Dict[str, DstcSchema],
    ):
        if not prev_tod_turn:
            context = ZsTodContext(max_length=self.cfg.num_turns)
            context.should_add_sys_actions = self.cfg.should_add_sys_actions
        else:
            context = copy.deepcopy(prev_tod_turn.context)
            context.system_utterances.append(
                prev_tod_turn.context.next_system_utterance
            )
            context.user_utterances.append(prev_tod_turn.context.current_user_utterance)
            context.prev_tod_turn = prev_tod_turn

        if user_turn:
            utterance = user_turn.utterance
            if self.cfg.delexicalize:
                utterance = self._delexicalize_utterance(user_turn, schemas)
            context.current_user_utterance = utterance
        if system_turn:
            utterance = system_turn.utterance
            if self.cfg.delexicalize:
                utterance = self._delexicalize_utterance(system_turn, schemas)
            context.next_system_utterance = utterance
            if self.cfg.should_add_service_results:
                if len(system_turn.frames) > 1:
                    raise ValueError("More than one frame in system turn")
                for frame in system_turn.frames:
                    context.service_results = frame.service_results
        return context

    def _prepare_dst(self, user_turn: DstcTurn) -> List[ZsTodBelief]:
        dsts = []
        for frame in user_turn.frames:
            if not frame.state:
                continue
            beliefs = []
            actions = []
            active_intent = frame.state.active_intent
            requested_slots = [
                "".join(
                    [
                        frame.short_service,
                        ZsTodConstants.DOMAIN_SLOT_SEPARATOR,
                        slot,
                    ]
                )
                for slot in frame.state.requested_slots
            ]
            for slot_name, value in frame.state.slot_values.items():
                beliefs.append(
                    ZsTodBelief(
                        frame.short_service,
                        # humps.camelize(slot_name),
                        slot_name,
                        value,
                    )
                )
            dsts.append(ZsTodDst(beliefs, active_intent, requested_slots))
        return dsts

    def _get_actions(self, turn: DstcTurn) -> list[ZsTodAction]:
        actions = []
        for frame in turn.frames:
            for action in frame.actions:
                actions.append(
                    ZsTodAction(
                        frame.short_service,
                        action.act,
                        action.slot,
                        ZsTodConstants.ACTION_VALUE_SEPARATOR.join(action.values),
                    )
                )
        return actions

    def _delexicalize_utterance(
        self, turn: DstcTurn, schemas: Dict[str, DstcSchema]
    ) -> str:
        delexicalized_utterance = turn.utterance
        for frame in turn.frames:
            schema = schemas[frame.short_service]
            for action in frame.actions:
                for value in action.values:
                    slot = next(
                        (slot for slot in schema.slots if slot.name == action.slot),
                        None,
                    )
                    if not slot:
                        continue
                    replacement = (
                        # f"<{frame.short_service}_{humps.camelize(action.slot)}>"
                        f"<{frame.short_service}{ZsTodConstants.DOMAIN_SLOT_SEPARATOR}{action.slot}>"
                    )
                    delexicalized_utterance = delexicalized_utterance.replace(
                        value, replacement
                    )
        return delexicalized_utterance

    def _prepare_response(
        self, system_turn: DstcTurn, schemas: Dict[str, DstcSchema]
    ) -> str:
        if not system_turn:
            return None
        if not self.cfg.delexicalize:
            return system_turn.utterance
        return self._delexicalize_utterance(system_turn, schemas)

    def _prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: Dict[str, DstcSchema],
    ) -> ZsTodTarget:
        dsts = self._prepare_dst(user_turn)
        actions = self._get_actions(system_turn)
        user_actions = self._get_actions(user_turn)
        response = self._prepare_response(system_turn, schemas)
        return ZsTodTarget(
            dsts=dsts, actions=actions, user_actions=user_actions, response=response
        )

    def _prepare_turn(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        prev_tod_turn: ZsTodTurn,
        schemas: Dict[str, DstcSchema],
        services: list[str],
    ) -> ZsTodTurn:
        turn_schemas = None
        turn_schema_str = None
        if self.cfg.should_add_schema:
            turn_schemas = [schemas[s] for s in services]
            turn_schema_str = "".join([str(s) for s in turn_schemas])
        context = self._prepare_context(user_turn, system_turn, prev_tod_turn, schemas)
        target = self._prepare_target(user_turn, system_turn, schemas)
        return ZsTodTurn(
            context, target, schemas=turn_schemas, schema_str=turn_schema_str
        )

    def _is_dialogue_in_domain(self, dialogue_services: List[str]) -> bool:
        return all(ds in self.cfg.domains for ds in dialogue_services)

    def _extract_from_target(
        self, target: str, start_tokens: list[str], end_tokens: list[str]
    ):
        texts = []
        for start_token, end_token in zip(start_tokens, end_tokens):
            try:
                start_index = target.index(start_token)
                end_index = target.index(end_token)
                texts.append(target[start_index : end_index + len(end_token)]),
            except ValueError:
                texts.append("")
        return "".join(
            [
                SpecialTokens.begin_target,
                "".join(texts),
                SpecialTokens.end_target,
            ]
        )

    def _get_schema_str(
        self,
        schemas: list[DstcSchema],
        turn: ZsTodTurn,
    ) -> str:
        if not schemas:
            return ""
        schema_str = "".join([schema.get_full_repr() for schema in turn.schemas])
        return schema_str

    def _prepare_dialog(
        self,
        dstc_dialog: DstcDialog,
        schemas: Dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> Optional[List[ZsTodTurn]]:
        tod_turns = []
        tod_turn = None
        if not self._is_dialogue_in_domain(dstc_dialog.services):
            return None

        for i, (user_turn, system_turn) in enumerate(
            utils.grouper(dstc_dialog.turns, 2)
        ):
            tod_turn = self._prepare_turn(
                user_turn, system_turn, tod_turn, schemas, dstc_dialog.services
            )
            tod_turn.dialog_id = dstc_dialog.dialogue_id
            tod_turn.turn_id = i + 1
            tod_turn.active_intent = user_turn.get_active_intent()

            tod_turns.append(
                turn_csv_row_handler.to_csv_row(
                    self.cfg.context_type, tod_turn, self.cfg.should_add_schema
                )
            )

        return tod_turns

    def _prepare_dialog_file(
        self,
        path: Path,
        schemas: Dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> np.ndarray:
        data = []
        dialog_json_data = utils.read_json(path)
        if any(x in dialog_json_data for x in ["~", "^"]):
            raise ValueError("dialog contains ~")
        for d in dialog_json_data:
            dialog = DstcDialog.from_dict(d)
            prepped_dialog = self._prepare_dialog(dialog, schemas, turn_csv_row_handler)
            if prepped_dialog is None:
                continue
            data.append(prepped_dialog)
        if not len(data):
            return np.array(data)
        return np.concatenate(data, axis=0)

    def run(self):
        steps = Steps.list()
        schemas = {}
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in steps]:
            schemas.update(d)
        turn_csv_row_handler: TurnCsvRowBase = GeneralTurnCsvRow()
        step_dir = Path(self.cfg.processed_data_root / self.cfg.step_name)
        step_dir.mkdir(parents=True, exist_ok=True)
        dialog_paths = get_dialog_file_paths(self.cfg.raw_data_root, self.cfg.step_name)
        out_data = []
        if self.cfg.num_dialogs == "None":
            self.cfg.num_dialogs = len(dialog_paths)
        csv_file_path = get_csv_data_path(
            step=self.cfg.step_name,
            num_dialogs=self.cfg.num_dialogs,
            cfg=self.cfg,
        )
        if csv_file_path.exists() and not self.cfg.overwrite:
            print(
                f"{self.cfg.step_name} csv file already exists and overwrite is false, so skipping"
            )
            return

        if self.cfg.data_prep_multi_process:
            res = list(
                tqdm(
                    Pool().imap(
                        self._prepare_dialog_file,
                        dialog_paths[: self.cfg.num_dialogs],
                        itertools.repeat(schemas),
                        itertools.repeat(turn_csv_row_handler),
                    ),
                    total=self.cfg.num_dialogs,
                )
            )
        # start no mp code
        else:
            res = []
            for d in tqdm(dialog_paths[: self.cfg.num_dialogs]):
                output = self._prepare_dialog_file(d, schemas, turn_csv_row_handler)
                if res is not None:
                    res.append(output)
        # end no mp code

        out_data = [d for d in res if len(d)]
        headers = turn_csv_row_handler.get_csv_headers(self.cfg.should_add_schema)
        if len(out_data) == 0:
            print(f"No data for {self.cfg.step_name}")
            return
        csv_data = np.concatenate(out_data, axis=0)
        utils.write_csv(headers, csv_data, csv_file_path)


@hydra.main(config_path="../config/data_prep/", config_name="zs_tod")
def hydra_start(cfg: DictConfig) -> None:
    stdp = ZsTodDSTCDataPrep(DataPrepConfig(**cfg))
    stdp.run()


if __name__ == "__main__":
    hydra_start()
