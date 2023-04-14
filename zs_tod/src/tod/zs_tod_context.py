

from collections import deque
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Optional

from my_enums import DstcSystemActions, SpecialTokens

@dataclass
class ZsTodContext:
    user_utterances: deque[str] = field(default_factory=deque)
    system_utterances: deque[str] = field(default_factory=deque)
    next_system_utterance: str = None
    current_user_utterance: str = None
    should_add_sys_actions: bool = None
    prev_tod_turn: Optional[any] = None
    service_results: Optional[list[dict[str, str]]] = None

    def __init__(self, max_length: int = 10):
        self.user_utterances = deque(maxlen=max_length)
        self.system_utterances = deque(maxlen=max_length)

    def __repr__(self) -> str:
        return self.__str__()

    def get_short_repr(self) -> str:
        return "".join(
            [
                SpecialTokens.begin_context,
                self.prev_tod_turn.target.get_dsts() if self.prev_tod_turn else "",
                self._get_service_results(),
                self._get_sys_actions(),
                self._get_last_user_utterance(),
                SpecialTokens.end_context,
            ]
        )

    def _get_service_results(self) -> str:
        out = ""
        if self.service_results:
            for service_result in self.service_results[:1]:
                out += "".join(
                    [
                        SpecialTokens.begin_service_results,
                        " ".join([" ".join([k, v]) for k, v in service_result.items()]),
                        SpecialTokens.end_service_results,
                    ]
                )
        return out

    def _get_sys_actions(self) -> str:
        if not self.should_add_sys_actions:
            return ""
        return "".join([SpecialTokens.sys_actions, " ".join(DstcSystemActions.list())])

    def _get_last_user_utterance(self) -> str:
        return "".join(
            [
                SpecialTokens.begin_last_user_utterance,
                self.current_user_utterance,
                SpecialTokens.end_last_user_utterance,
            ]
        )

    def __str__(self) -> str:
        out = SpecialTokens.begin_context
        for user, system in zip_longest(
            self.user_utterances, self.system_utterances, fillvalue=""
        ):
            if user:
                out += SpecialTokens.user + user
            if system:
                out += SpecialTokens.system + system

        out += self._get_service_results()

        out += self._get_sys_actions()
        out += self._get_last_user_utterance()
        out += SpecialTokens.end_context
        return out
