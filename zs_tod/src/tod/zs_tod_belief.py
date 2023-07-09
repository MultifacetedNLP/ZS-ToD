from dataclasses import dataclass
from typing import Optional

from my_enums import ZsTodConstants
import utils


@dataclass
class ZsTodBelief:
    domain: str
    slot_name: str
    values: any
    prediction: Optional[str] = ""
    is_categorical: Optional[bool] = None

    @classmethod
    def from_string(
        self, text: str, slot_categories: dict[str, bool] = None
    ) -> "ZsTodBelief":
        try:
            dom_slot, values_str = text.split(ZsTodConstants.SLOT_VALUE_SEPARATOR)
            values = values_str.split(ZsTodConstants.VALUE_SEPARATOR)
        except ValueError:
            return self("", "", "", text)
        try:
            domain, slot_name = dom_slot.split(ZsTodConstants.DOMAIN_SLOT_SEPARATOR)
        except ValueError:
            return self("", "", values, text)
        is_categorical = None
        if slot_categories:
            is_categorical = slot_categories[slot_name]
        return self(domain, slot_name, values, is_categorical=is_categorical)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "".join(
            [
                self.domain,
                ZsTodConstants.DOMAIN_SLOT_SEPARATOR,
                self.slot_name,
                ZsTodConstants.SLOT_VALUE_SEPARATOR,
                ZsTodConstants.VALUE_SEPARATOR.join(self.values),
            ]
        )

    def __eq__(self, other: "ZsTodBelief") -> bool:
        return (
            self.domain == other.domain
            and self.slot_name == other.slot_name
            and utils.get_slot_value_match_score(
                self.values, other.values, self.is_categorical
            )
        )
