from typing import Iterable, Optional, Sequence

import pandas as pd

from research_payments.models import PaymentSchema, PhysicianQuery


def detect_schema(columns: Iterable[str]) -> PaymentSchema:
    available = set(columns)
    if {"Physician_First_Name", "Physician_Last_Name"}.issubset(available):
        return PaymentSchema(
            first_name_col="Physician_First_Name",
            middle_name_col="Physician_Middle_Name",
            last_name_col="Physician_Last_Name",
            npi_col="Physician_NPI",
            profile_id_col="Physician_Profile_ID",
            specialty_prefix="Physician_Specialty",
        )

    return PaymentSchema(
        first_name_col="Covered_Recipient_First_Name",
        middle_name_col="Covered_Recipient_Middle_Name",
        last_name_col="Covered_Recipient_Last_Name",
        npi_col="Covered_Recipient_NPI",
        profile_id_col="Covered_Recipient_Profile_ID",
        specialty_prefix="Covered_Recipient_Specialty",
    )


def validate_columns(columns: Iterable[str], required: Sequence[str]) -> None:
    available = set(columns)
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")


def normalize_text(value: Optional[str], case_sensitive: bool) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if case_sensitive else normalized.lower()


def normalize_series(series: pd.Series, case_sensitive: bool) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip()
    return normalized if case_sensitive else normalized.str.lower()


def is_middle_initial(value: str) -> bool:
    return len(value.replace(".", "").strip()) == 1


def middle_name_mask(
    series: pd.Series,
    middle_name: Optional[str],
    case_sensitive: bool,
) -> pd.Series:
    if not middle_name:
        return pd.Series(True, index=series.index)

    normalized_series = normalize_series(series, case_sensitive)
    normalized_middle = normalize_text(middle_name, case_sensitive)
    assert normalized_middle is not None

    if is_middle_initial(normalized_middle):
        return normalized_series.str.startswith(normalized_middle.replace(".", ""))

    return normalized_series.eq(normalized_middle)


def name_match_mask(
    df: pd.DataFrame,
    query: PhysicianQuery,
    first_col: str,
    last_col: str,
    middle_col: Optional[str],
) -> pd.Series:
    first_mask = normalize_series(df[first_col], query.case_sensitive).eq(
        normalize_text(query.first_name, query.case_sensitive)
    )
    last_mask = normalize_series(df[last_col], query.case_sensitive).eq(
        normalize_text(query.last_name, query.case_sensitive)
    )

    if query.middle_name and middle_col and middle_col in df.columns:
        return first_mask & last_mask & middle_name_mask(
            df[middle_col],
            query.middle_name,
            query.case_sensitive,
        )

    return first_mask & last_mask
