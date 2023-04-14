

from abc import ABC, abstractmethod

from my_enums import ContextType
from tod.turns.zs_tod_turn import ZsTodTurn


class TurnCsvRowBase(ABC):

    @abstractmethod
    def get_csv_headers(self, should_add_schema: bool)->list[str]:
        headers= ["dialog_id", "turn_id", "context"]
        if should_add_schema:
            headers.append("schema")
        return headers
    
    @abstractmethod
    def to_csv_row(self, context_type:ContextType, tod_turn: ZsTodTurn, should_add_schema: bool)->list[str]:
        context_str = (
            str(tod_turn.context)
            if context_type == ContextType.DEFAULT
            else tod_turn.context.get_short_repr()
        )
        context_str += tod_turn.prompt_token if tod_turn.prompt_token else ""
        out =  [tod_turn.dialog_id, tod_turn.turn_id, context_str]
        if should_add_schema:
            out.append(tod_turn.schema_str)
        return out


