import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from research_payments.dataset import OpenPaymentsDataset
from research_payments.models import PhysicianQuery, PhysicianResult


logger = logging.getLogger(__name__)


class ResearchPaymentsProcessor:
    def process_physician(
        self,
        first_name: str,
        middle_name: Optional[str],
        last_name: str,
        years_to_process: Sequence[int],
        case_sensitive: bool = False,
    ) -> PhysicianResult:
        query = PhysicianQuery(
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            case_sensitive=case_sensitive,
        )
        yearly_results: Dict[int, pd.DataFrame] = {}

        for year in years_to_process:
            csv_path = Path(f"{year}_rsh_payments.csv")
            yearly_df = self._process_year(csv_path, query)
            if yearly_df is not None and not yearly_df.empty:
                yearly_results[year] = yearly_df

        return self._combine_yearly_results(yearly_results, years_to_process)

    def _process_year(
        self, csv_path: Path, query: PhysicianQuery
    ) -> Optional[pd.DataFrame]:
        dataset = OpenPaymentsDataset(csv_path)
        search_queries = [query]
        if query.middle_name:
            search_queries.append(
                PhysicianQuery(
                    first_name=query.first_name,
                    last_name=query.last_name,
                    middle_name=None,
                    case_sensitive=query.case_sensitive,
                )
            )

        frames = []
        for search_query in search_queries:
            try:
                result = dataset.search(search_query)
            except FileNotFoundError:
                raise
            except Exception as exc:
                logger.error(
                    "Error searching %s in %s: %s",
                    search_query.display_name,
                    csv_path,
                    exc,
                )
                return None

            if not result.dataframe.empty:
                frames.append(result.dataframe)

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.groupby(["NPI", "Physician_Name", "Specialty"], as_index=False)
            .agg(
                Entry_Count=("Entry_Count", "max"),
                Total_Payment_USD=("Total_Payment_USD", "max"),
            )
            .sort_values("Total_Payment_USD", ascending=False)
            .reset_index(drop=True)
        )
        return combined

    def _combine_yearly_results(
        self,
        yearly_results: Dict[int, pd.DataFrame],
        years_to_process: Sequence[int],
    ) -> PhysicianResult:
        if not yearly_results:
            return PhysicianResult(dataframe=None, total_payment=0.0, total_entries=0)

        frames = []
        for year, year_df in yearly_results.items():
            frames.append(
                year_df.rename(columns={"Total_Payment_USD": f"Payment_{year}_USD"})
            )

        combined_df = pd.concat(frames, ignore_index=True, sort=False)
        payment_columns = [f"Payment_{year}_USD" for year in years_to_process]
        for column in payment_columns:
            if column not in combined_df.columns:
                combined_df[column] = 0.0
            combined_df[column] = pd.to_numeric(
                combined_df[column], errors="coerce"
            ).fillna(0.0)

        combined_df["Entry_Count"] = (
            pd.to_numeric(combined_df["Entry_Count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        combined_df = (
            combined_df.groupby("NPI", as_index=False)
            .agg(
                Physician_Name=("Physician_Name", "first"),
                Specialty=("Specialty", "first"),
                Entry_Count=("Entry_Count", "sum"),
                **{column: (column, "sum") for column in payment_columns},
            )
            .reset_index(drop=True)
        )
        combined_df["Total_USD"] = combined_df[payment_columns].sum(axis=1)
        combined_df = combined_df.sort_values("Total_USD", ascending=False)
        combined_df = combined_df[
            [
                "NPI",
                "Physician_Name",
                "Specialty",
                "Entry_Count",
                *payment_columns,
                "Total_USD",
            ]
        ].reset_index(drop=True)

        return PhysicianResult(
            dataframe=combined_df,
            total_payment=float(combined_df["Total_USD"].sum()),
            total_entries=int(combined_df["Entry_Count"].sum()),
        )
