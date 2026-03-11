import re
from typing import List, Optional, Sequence, Tuple

import pandas as pd


def parse_name(full_name: str) -> Tuple[Optional[str], Optional[str], str]:
    cleaned = re.sub(r"\([^)]*\)", "", full_name).strip()
    parts = cleaned.split()

    if len(parts) == 1:
        return None, None, parts[0]
    if len(parts) == 2:
        return parts[0], None, parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return parts[0], " ".join(parts[1:-1]), parts[-1]


def read_names_from_file(filename: str) -> List[str]:
    with open(filename, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def year_range_label(years: Sequence[int]) -> str:
    return f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])


def safe_filename(value: str) -> str:
    return re.sub(r"[^\w\s-]", "", value).strip().replace(" ", "_")


def format_currency_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        formatted[column] = pd.to_numeric(formatted[column], errors="coerce").fillna(0.0)
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
    return formatted
