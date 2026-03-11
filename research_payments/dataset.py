import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from research_payments.constants import PAYMENT_COLUMN, PI_SLOTS
from research_payments.matching import detect_schema, name_match_mask, validate_columns
from research_payments.models import PaymentSchema, PhysicianQuery, SearchResult


logger = logging.getLogger(__name__)


class OpenPaymentsDataset:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.schema: PaymentSchema | None = None
        self.dataframe: pd.DataFrame | None = None

    def load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        logger.info("Loading %s", self.csv_path)
        sample_df = pd.read_csv(self.csv_path, nrows=5, dtype=str)
        self.schema = detect_schema(sample_df.columns)
        validate_columns(
            sample_df.columns,
            [PAYMENT_COLUMN, self.schema.first_name_col, self.schema.last_name_col],
        )

        df = pd.read_csv(self.csv_path, dtype=str, low_memory=False)
        df[PAYMENT_COLUMN] = pd.to_numeric(df[PAYMENT_COLUMN], errors="coerce")

        invalid_rows = int(df[PAYMENT_COLUMN].isna().sum())
        if invalid_rows:
            logger.warning(
                "Dropping %s rows with invalid payment amounts in %s",
                invalid_rows,
                self.csv_path,
            )
            df = df.dropna(subset=[PAYMENT_COLUMN]).copy()

        self.dataframe = df

    def search(self, query: PhysicianQuery) -> SearchResult:
        if self.dataframe is None or self.schema is None:
            self.load()

        assert self.dataframe is not None
        assert self.schema is not None

        covered_mask = name_match_mask(
            self.dataframe,
            query,
            self.schema.first_name_col,
            self.schema.last_name_col,
            self.schema.middle_name_col,
        )

        pi_mask = pd.Series(False, index=self.dataframe.index)
        for slot in PI_SLOTS:
            pi_first_col = f"{self.schema.pi_prefix}{slot}_First_Name"
            pi_last_col = f"{self.schema.pi_prefix}{slot}_Last_Name"
            pi_middle_col = f"{self.schema.pi_prefix}{slot}_Middle_Name"
            if (
                pi_first_col in self.dataframe.columns
                and pi_last_col in self.dataframe.columns
            ):
                pi_mask |= name_match_mask(
                    self.dataframe,
                    query,
                    pi_first_col,
                    pi_last_col,
                    pi_middle_col if pi_middle_col in self.dataframe.columns else None,
                )

        matched_df = self.dataframe.loc[covered_mask | pi_mask].copy()
        logger.info(
            "Found %s matching rows for %s in %s",
            len(matched_df),
            query.display_name,
            self.csv_path.name,
        )
        if matched_df.empty:
            return SearchResult(pd.DataFrame(), {})

        return self._summarize_matches(matched_df)

    def _summarize_matches(self, matched_df: pd.DataFrame) -> SearchResult:
        matched_df = matched_df.copy()
        matched_df["NPI"] = matched_df.apply(self._resolve_identifier, axis=1)

        totals_df = (
            matched_df.groupby("NPI", as_index=False)[PAYMENT_COLUMN]
            .sum()
            .rename(columns={PAYMENT_COLUMN: "Total_Payment_USD"})
        )
        entry_counts = matched_df["NPI"].value_counts().to_dict()

        metadata_rows = []
        for npi in totals_df["NPI"]:
            source_row = matched_df.loc[matched_df["NPI"] == npi].iloc[0]
            physician_name, specialty = self._extract_identity(source_row)
            metadata_rows.append(
                {
                    "NPI": npi,
                    "Physician_Name": physician_name,
                    "Specialty": specialty,
                    "Entry_Count": int(entry_counts[npi]),
                }
            )

        metadata_df = pd.DataFrame(metadata_rows)
        result_df = totals_df.merge(metadata_df, on="NPI", how="left")
        result_df = result_df[
            ["NPI", "Physician_Name", "Specialty", "Entry_Count", "Total_Payment_USD"]
        ].sort_values("Total_Payment_USD", ascending=False)
        result_df.reset_index(drop=True, inplace=True)

        return SearchResult(result_df, entry_counts)

    def _resolve_identifier(self, row: pd.Series) -> str:
        assert self.schema is not None

        direct_npi = self._clean_cell(row.get(self.schema.npi_col))
        if direct_npi:
            return direct_npi

        for slot in PI_SLOTS:
            pi_npi = self._clean_cell(row.get(f"{self.schema.pi_prefix}{slot}_NPI"))
            if pi_npi:
                return pi_npi

        profile_id = self._clean_cell(row.get(self.schema.profile_id_col))
        if profile_id:
            return f"PROFILE_{profile_id}"

        return f"UNKNOWN_{row.name}"

    def _extract_identity(self, row: pd.Series) -> Tuple[str, str]:
        assert self.schema is not None

        covered_first = self._clean_cell(row.get(self.schema.first_name_col))
        covered_last = self._clean_cell(row.get(self.schema.last_name_col))
        if covered_first and covered_last:
            covered_middle = self._clean_cell(row.get(self.schema.middle_name_col))
            return self._join_name(
                covered_first, covered_middle, covered_last
            ), self._extract_specialty(row)

        for slot in PI_SLOTS:
            first_name = self._clean_cell(row.get(f"{self.schema.pi_prefix}{slot}_First_Name"))
            middle_name = self._clean_cell(row.get(f"{self.schema.pi_prefix}{slot}_Middle_Name"))
            last_name = self._clean_cell(row.get(f"{self.schema.pi_prefix}{slot}_Last_Name"))
            if first_name and last_name:
                return (
                    self._join_name(first_name, middle_name, last_name),
                    "Principal Investigator",
                )

        return "Unknown", "Unknown"

    def _extract_specialty(self, row: pd.Series) -> str:
        assert self.schema is not None

        specialties = []
        for slot in PI_SLOTS:
            specialty = self._clean_cell(row.get(f"{self.schema.specialty_prefix}_{slot}"))
            if specialty:
                specialties.append(specialty)

        if specialties:
            return "; ".join(specialties)

        fallback = self._clean_cell(row.get(self.schema.specialty_prefix))
        return fallback or "Unknown"

    @staticmethod
    def _join_name(first_name: str, middle_name: str, last_name: str) -> str:
        return " ".join(
            part for part in [first_name, middle_name, last_name] if part
        ).strip()

    @staticmethod
    def _clean_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()
