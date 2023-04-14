from dataclasses import dataclass
from typing import Optional
from sgd_dstc8_data_model.dstc_dataclasses import DstcSchema
from my_enums import SpecialTokens

from tod.zs_target import ZsTodTarget
from tod.zs_tod_context import ZsTodContext


@dataclass
class TodTurnCsvRow:
    dialog_id: str
    turn_id: str
    context: str
    target: str = None
    schema: Optional[str] = None


@dataclass
class ZsTodTurn:
    context: ZsTodContext
    target: ZsTodTarget
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None
    schemas: Optional[list[DstcSchema]] = None
    active_intent: Optional[str] = None
    schema_str: Optional[str] = None
    prompt_token: Optional[SpecialTokens] = None
