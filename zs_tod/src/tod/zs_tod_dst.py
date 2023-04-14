from dataclasses import dataclass
from typing import Optional
from sgd_dstc8_data_model.dstc_dataclasses import DstcRequestedSlot

from my_enums import SimpleTodConstants, SpecialTokens
from tod.zs_tod_belief import ZsTodBelief


@dataclass
class ZsTodDst:
    beliefs: list[ZsTodBelief]
    active_intent: str
    requested_slots: Optional[list[DstcRequestedSlot]] = None

    def get_belief_repr(self) -> str:
        return SimpleTodConstants.ITEM_SEPARATOR.join(map(str, self.beliefs))

    def get_req_slots_str(self) -> str:
        return SimpleTodConstants.ITEM_SEPARATOR.join(map(str, self.requested_slots))

    def __str__(self) -> str:
        intents_str = (
            "".join(
                [
                    SpecialTokens.begin_intent,
                    self.active_intent,
                    SpecialTokens.end_intent,
                ]
            )
            if self.active_intent
            else ""
        )

        slots_str = (
            "".join(
                [
                    SpecialTokens.begin_requested_slots,
                    self.get_req_slots_str(),
                    SpecialTokens.end_requested_slots,
                ]
            )
            if self.requested_slots
            else ""
        )
        out = "".join(
            [
                SpecialTokens.begin_dst,
                intents_str,
                slots_str,
                SpecialTokens.begin_belief,
                self.get_belief_repr(),
                SpecialTokens.end_belief,
                SpecialTokens.end_dst,
            ]
        )
        return out
