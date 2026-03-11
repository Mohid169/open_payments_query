from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class PhysicianQuery:
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    case_sensitive: bool = False

    @property
    def display_name(self) -> str:
        return " ".join(
            part for part in [self.first_name, self.middle_name or "", self.last_name] if part
        ).strip()


@dataclass(frozen=True)
class PaymentSchema:
    first_name_col: str
    middle_name_col: str
    last_name_col: str
    npi_col: str
    profile_id_col: str
    specialty_prefix: str
    pi_prefix: str = "Principal_Investigator_"


@dataclass
class SearchResult:
    dataframe: pd.DataFrame
    entry_counts: Dict[str, int]


@dataclass
class PhysicianResult:
    dataframe: Optional[pd.DataFrame]
    total_payment: float
    total_entries: int
