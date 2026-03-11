import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from research_payments.constants import PAYMENT_COLUMN, PI_SLOTS
from research_payments.matching import detect_schema
from research_payments.models import PaymentSchema


def build_searchable_records_from_zip(zip_path: Path) -> List[Dict[str, object]]:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    records: List[Dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.endswith("/") or not member.lower().endswith(".csv"):
                continue

            with archive.open(member) as handle:
                raw_bytes = handle.read()

            df = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, low_memory=False)
            if PAYMENT_COLUMN not in df.columns:
                continue

            schema = detect_schema(df.columns)
            year = extract_year_from_name(member)
            records.extend(_records_from_dataframe(df, schema, year, member))

    return aggregate_records(records)


def aggregate_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not records:
        return []

    df = pd.DataFrame(records)
    grouped = (
        df.groupby(
            [
                "display_name",
                "first_name",
                "middle_name",
                "last_name",
                "role",
                "year",
                "identifier",
                "identifier_type",
                "specialty",
                "source_file",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            total_payment=("payment_amount", "sum"),
            instance_count=("payment_amount", "size"),
        )
        .sort_values(
            ["display_name", "year", "role", "total_payment"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )
    return grouped.to_dict(orient="records")


def extract_year_from_name(name: str) -> str:
    match = re.search(r"(19|20)\d{2}", name)
    return match.group(0) if match else "Unknown"


def _records_from_dataframe(
    df: pd.DataFrame,
    schema: PaymentSchema,
    year: str,
    source_file: str,
) -> List[Dict[str, object]]:
    work_df = df.copy()
    work_df[PAYMENT_COLUMN] = pd.to_numeric(work_df[PAYMENT_COLUMN], errors="coerce")
    work_df = work_df.dropna(subset=[PAYMENT_COLUMN])

    records: List[Dict[str, object]] = []
    records.extend(_covered_records(work_df, schema, year, source_file))
    records.extend(_pi_records(work_df, schema, year, source_file))
    return records


def _covered_records(
    df: pd.DataFrame,
    schema: PaymentSchema,
    year: str,
    source_file: str,
) -> List[Dict[str, object]]:
    records = []
    required_columns = [schema.first_name_col, schema.last_name_col]
    available = df.dropna(subset=required_columns)

    for _, row in available.iterrows():
        first_name = _clean_cell(row.get(schema.first_name_col))
        last_name = _clean_cell(row.get(schema.last_name_col))
        if not first_name or not last_name:
            continue

        middle_name = _clean_cell(row.get(schema.middle_name_col))
        identifier, identifier_type = _resolve_covered_identifier(row, schema)
        specialty = _extract_specialty(row, schema)
        records.append(
            _build_record(
                first_name=first_name,
                middle_name=middle_name,
                last_name=last_name,
                role="Covered Recipient",
                year=year,
                source_file=source_file,
                payment_amount=float(row[PAYMENT_COLUMN]),
                identifier=identifier,
                identifier_type=identifier_type,
                specialty=specialty,
            )
        )

    return records


def _pi_records(
    df: pd.DataFrame,
    schema: PaymentSchema,
    year: str,
    source_file: str,
) -> List[Dict[str, object]]:
    records = []
    for _, row in df.iterrows():
        for slot in PI_SLOTS:
            first_name = _clean_cell(row.get(f"{schema.pi_prefix}{slot}_First_Name"))
            last_name = _clean_cell(row.get(f"{schema.pi_prefix}{slot}_Last_Name"))
            if not first_name or not last_name:
                continue

            middle_name = _clean_cell(row.get(f"{schema.pi_prefix}{slot}_Middle_Name"))
            identifier = _clean_cell(row.get(f"{schema.pi_prefix}{slot}_NPI"))
            records.append(
                _build_record(
                    first_name=first_name,
                    middle_name=middle_name,
                    last_name=last_name,
                    role=f"Principal Investigator {slot}",
                    year=year,
                    source_file=source_file,
                    payment_amount=float(row[PAYMENT_COLUMN]),
                    identifier=identifier,
                    identifier_type="NPI" if identifier else "Missing",
                    specialty="Principal Investigator",
                )
            )
    return records


def _build_record(
    *,
    first_name: str,
    middle_name: str,
    last_name: str,
    role: str,
    year: str,
    source_file: str,
    payment_amount: float,
    identifier: str,
    identifier_type: str,
    specialty: str,
) -> Dict[str, object]:
    display_name = " ".join(part for part in [first_name, middle_name, last_name] if part).strip()
    return {
        "display_name": display_name,
        "display_name_normalized": display_name.lower(),
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "role": role,
        "year": year,
        "source_file": source_file,
        "payment_amount": payment_amount,
        "identifier": identifier,
        "identifier_type": identifier_type,
        "specialty": specialty or "Unknown",
    }


def _resolve_covered_identifier(row: pd.Series, schema: PaymentSchema) -> tuple[str, str]:
    npi = _clean_cell(row.get(schema.npi_col))
    if npi:
        return npi, "NPI"

    profile_id = _clean_cell(row.get(schema.profile_id_col))
    if profile_id:
        return profile_id, "Profile ID"

    return "", "Missing"


def _extract_specialty(row: pd.Series, schema: PaymentSchema) -> str:
    specialties = []
    for slot in PI_SLOTS:
        specialty = _clean_cell(row.get(f"{schema.specialty_prefix}_{slot}"))
        if specialty:
            specialties.append(specialty)
    if specialties:
        return "; ".join(specialties)

    return _clean_cell(row.get(schema.specialty_prefix)) or "Unknown"


def _clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()
