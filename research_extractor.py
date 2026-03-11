import argparse
import json
import logging
import os
import re
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tabulate import tabulate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PAYMENT_COLUMN = "Total_Amount_of_Payment_USDollars"
PI_SLOT_RANGE = range(1, 6)


@dataclass(frozen=True)
class PhysicianQuery:
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    case_sensitive: bool = False

    @property
    def display_name(self) -> str:
        parts = [self.first_name, self.middle_name or "", self.last_name]
        return " ".join(part for part in parts if part).strip()


@dataclass(frozen=True)
class PaymentCsvSchema:
    first_name_col: str
    middle_name_col: str
    last_name_col: str
    npi_col: str
    profile_id_col: str
    specialty_prefix: str
    pi_prefix: str = "Principal_Investigator_"


@dataclass
class YearProcessingResult:
    dataframe: Optional[pd.DataFrame]
    total_payment: float
    total_entries: int


def detect_schema(columns: Iterable[str]) -> PaymentCsvSchema:
    column_set = set(columns)
    if {"Physician_First_Name", "Physician_Last_Name"}.issubset(column_set):
        logger.info("Detected pre-2016 column format")
        return PaymentCsvSchema(
            first_name_col="Physician_First_Name",
            middle_name_col="Physician_Middle_Name",
            last_name_col="Physician_Last_Name",
            npi_col="Physician_NPI",
            profile_id_col="Physician_Profile_ID",
            specialty_prefix="Physician_Specialty",
        )

    logger.info("Detected 2016+ column format")
    return PaymentCsvSchema(
        first_name_col="Covered_Recipient_First_Name",
        middle_name_col="Covered_Recipient_Middle_Name",
        last_name_col="Covered_Recipient_Last_Name",
        npi_col="Covered_Recipient_NPI",
        profile_id_col="Covered_Recipient_Profile_ID",
        specialty_prefix="Covered_Recipient_Specialty",
    )


def require_columns(columns: Iterable[str], required: Sequence[str]) -> None:
    column_set = set(columns)
    missing = [column for column in required if column not in column_set]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")


def normalize_series(series: pd.Series, case_sensitive: bool) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip()
    return normalized if case_sensitive else normalized.str.lower()


def normalize_value(value: Optional[str], case_sensitive: bool) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if case_sensitive else normalized.lower()


def is_middle_initial(value: str) -> bool:
    cleaned = value.replace(".", "").strip()
    return len(cleaned) == 1


def middle_name_mask(
    series: pd.Series,
    requested_middle: Optional[str],
    case_sensitive: bool,
) -> pd.Series:
    if not requested_middle:
        return pd.Series(True, index=series.index)

    normalized_series = normalize_series(series, case_sensitive)
    normalized_middle = normalize_value(requested_middle, case_sensitive)
    assert normalized_middle is not None

    if is_middle_initial(normalized_middle):
        initial = normalized_middle.replace(".", "")
        return normalized_series.str.startswith(initial)

    return normalized_series.eq(normalized_middle)


def name_match_mask(
    df: pd.DataFrame,
    first_col: str,
    last_col: str,
    middle_col: Optional[str],
    query: PhysicianQuery,
) -> pd.Series:
    first_mask = normalize_series(df[first_col], query.case_sensitive).eq(
        normalize_value(query.first_name, query.case_sensitive)
    )
    last_mask = normalize_series(df[last_col], query.case_sensitive).eq(
        normalize_value(query.last_name, query.case_sensitive)
    )

    if query.middle_name and middle_col and middle_col in df.columns:
        return first_mask & last_mask & middle_name_mask(
            df[middle_col],
            query.middle_name,
            query.case_sensitive,
        )

    return first_mask & last_mask


def choose_identifier(row: pd.Series, schema: PaymentCsvSchema) -> str:
    direct_npi = str(row.get(schema.npi_col, "") or "").strip()
    if direct_npi:
        return direct_npi

    for slot in PI_SLOT_RANGE:
        pi_npi_col = f"{schema.pi_prefix}{slot}_NPI"
        pi_npi = str(row.get(pi_npi_col, "") or "").strip()
        if pi_npi:
            return pi_npi

    profile_id = str(row.get(schema.profile_id_col, "") or "").strip()
    if profile_id:
        return f"PROFILE_{profile_id}"

    return f"UNKNOWN_{row.name}"


def extract_specialty(row: pd.Series, schema: PaymentCsvSchema) -> str:
    specialties: List[str] = []
    for slot in PI_SLOT_RANGE:
        specialty_col = f"{schema.specialty_prefix}_{slot}"
        specialty = str(row.get(specialty_col, "") or "").strip()
        if specialty:
            specialties.append(specialty)

    if specialties:
        return "; ".join(specialties)

    fallback_specialty = str(row.get(schema.specialty_prefix, "") or "").strip()
    return fallback_specialty or "Unknown"


def extract_display_name(row: pd.Series, schema: PaymentCsvSchema) -> Tuple[str, str]:
    covered_first = str(row.get(schema.first_name_col, "") or "").strip()
    covered_last = str(row.get(schema.last_name_col, "") or "").strip()

    if covered_first and covered_last:
        covered_middle = str(row.get(schema.middle_name_col, "") or "").strip()
        parts = [covered_first, covered_middle, covered_last]
        return " ".join(part for part in parts if part).strip(), extract_specialty(
            row, schema
        )

    for slot in PI_SLOT_RANGE:
        first_col = f"{schema.pi_prefix}{slot}_First_Name"
        middle_col = f"{schema.pi_prefix}{slot}_Middle_Name"
        last_col = f"{schema.pi_prefix}{slot}_Last_Name"
        pi_first = str(row.get(first_col, "") or "").strip()
        pi_last = str(row.get(last_col, "") or "").strip()
        if pi_first and pi_last:
            pi_middle = str(row.get(middle_col, "") or "").strip()
            parts = [pi_first, pi_middle, pi_last]
            return " ".join(part for part in parts if part).strip(), "Principal Investigator"

    return "Unknown", "Unknown"


def summarize_matches(
    matched_df: pd.DataFrame,
    schema: PaymentCsvSchema,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    matched = matched_df.copy()
    matched["NPI"] = matched.apply(lambda row: choose_identifier(row, schema), axis=1)

    grouped = (
        matched.groupby("NPI", as_index=False)[PAYMENT_COLUMN]
        .sum()
        .rename(columns={PAYMENT_COLUMN: "Total_Payment_USD"})
    )
    entry_counts = matched["NPI"].value_counts().to_dict()

    metadata_rows = []
    for npi in grouped["NPI"]:
        row = matched.loc[matched["NPI"] == npi].iloc[0]
        physician_name, specialty = extract_display_name(row, schema)
        metadata_rows.append(
            {
                "NPI": npi,
                "Physician_Name": physician_name,
                "Specialty": specialty,
                "Entry_Count": entry_counts[npi],
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    result = grouped.merge(metadata_df, on="NPI", how="left")
    result = result[
        ["NPI", "Physician_Name", "Specialty", "Entry_Count", "Total_Payment_USD"]
    ].sort_values("Total_Payment_USD", ascending=False)

    return result.reset_index(drop=True), entry_counts


def get_research_payments_for_physician(
    csv_path: str,
    physician_last_name: str,
    physician_first_name: str,
    physician_middle: Optional[str] = None,
    case_sensitive: bool = False,
    strict_middle_name: bool = False,
):
    del strict_middle_name

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Reading CMS Open Payments file: %s", csv_path)
    sample_df = pd.read_csv(csv_file, nrows=5, dtype=str)
    schema = detect_schema(sample_df.columns)
    require_columns(
        sample_df.columns,
        [PAYMENT_COLUMN, schema.first_name_col, schema.last_name_col],
    )

    df = pd.read_csv(csv_file, dtype=str, low_memory=False)
    df[PAYMENT_COLUMN] = pd.to_numeric(df[PAYMENT_COLUMN], errors="coerce")

    invalid_payment_rows = df[PAYMENT_COLUMN].isna().sum()
    if invalid_payment_rows:
        logger.warning("Dropped %s rows with invalid payment amounts", invalid_payment_rows)
        df = df.dropna(subset=[PAYMENT_COLUMN]).copy()

    query = PhysicianQuery(
        first_name=physician_first_name,
        middle_name=physician_middle,
        last_name=physician_last_name,
        case_sensitive=case_sensitive,
    )

    covered_mask = name_match_mask(
        df,
        schema.first_name_col,
        schema.last_name_col,
        schema.middle_name_col,
        query,
    )

    pi_mask = pd.Series(False, index=df.index)
    for slot in PI_SLOT_RANGE:
        pi_first_col = f"{schema.pi_prefix}{slot}_First_Name"
        pi_last_col = f"{schema.pi_prefix}{slot}_Last_Name"
        pi_middle_col = f"{schema.pi_prefix}{slot}_Middle_Name"
        if pi_first_col in df.columns and pi_last_col in df.columns:
            pi_mask |= name_match_mask(
                df,
                pi_first_col,
                pi_last_col,
                pi_middle_col if pi_middle_col in df.columns else None,
                query,
            )

    filtered_df = df.loc[covered_mask | pi_mask].copy()
    logger.info("Found %s entries matching '%s'", len(filtered_df), query.display_name)

    if filtered_df.empty:
        logger.warning("No payments found for %s", query.display_name)
        return 0.0

    return summarize_matches(filtered_df, schema)


def parse_name(full_name: str) -> Tuple[Optional[str], Optional[str], str]:
    cleaned_name = re.sub(r"\([^)]*\)", "", full_name).strip()
    parts = cleaned_name.split()

    if len(parts) == 1:
        return None, None, parts[0]
    if len(parts) == 2:
        return parts[0], None, parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]

    return parts[0], " ".join(parts[1:-1]), parts[-1]


def read_names_from_file(filename: str) -> List[str]:
    try:
        with open(filename, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
        logger.info("Successfully read %s names from %s", len(names), filename)
        return names
    except OSError as exc:
        logger.error("Error reading names file: %s", exc)
        return []


def merge_year_results(results_by_year: Dict[int, YearProcessingResult]) -> YearProcessingResult:
    detailed_years = {
        year: result
        for year, result in results_by_year.items()
        if result.dataframe is not None and not result.dataframe.empty
    }

    if not detailed_years:
        total_payment = sum(result.total_payment for result in results_by_year.values())
        return YearProcessingResult(dataframe=None, total_payment=total_payment, total_entries=0)

    dataframes = []
    for year, result in detailed_years.items():
        year_df = result.dataframe.copy()
        year_df = year_df.rename(columns={"Total_Payment_USD": f"Payment_{year}_USD"})
        dataframes.append(year_df)

    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    payment_columns = [column for column in combined_df.columns if column.startswith("Payment_")]
    for column in payment_columns:
        combined_df[column] = pd.to_numeric(combined_df[column], errors="coerce").fillna(0.0)

    combined_df["Entry_Count"] = pd.to_numeric(combined_df["Entry_Count"], errors="coerce").fillna(0).astype(int)
    combined_df = (
        combined_df.groupby("NPI", as_index=False)
        .agg(
            Physician_Name=("Physician_Name", "first"),
            Specialty=("Specialty", "first"),
            Entry_Count=("Entry_Count", "sum"),
            **{column: (column, "sum") for column in payment_columns},
        )
    )
    combined_df["Total_USD"] = combined_df[payment_columns].sum(axis=1)
    combined_df = combined_df.sort_values("Total_USD", ascending=False).reset_index(drop=True)

    total_payment = float(combined_df["Total_USD"].sum())
    total_entries = int(combined_df["Entry_Count"].sum())
    return YearProcessingResult(combined_df, total_payment, total_entries)


def process_physician(
    first_name: str,
    middle_name: Optional[str],
    last_name: str,
    years_to_process: Sequence[int],
    case_sensitive: bool = False,
) -> Tuple[Optional[pd.DataFrame], float, int]:
    results_by_year: Dict[int, YearProcessingResult] = {}

    for year in years_to_process:
        csv_file_path = f"{year}_rsh_payments.csv"
        logger.info("Processing year %s from %s", year, csv_file_path)

        try:
            search_results = []
            if middle_name:
                search_results.append(
                    get_research_payments_for_physician(
                        csv_file_path,
                        last_name,
                        first_name,
                        physician_middle=middle_name,
                        case_sensitive=case_sensitive,
                        strict_middle_name=True,
                    )
                )

            search_results.append(
                get_research_payments_for_physician(
                    csv_file_path,
                    last_name,
                    first_name,
                    physician_middle=None,
                    case_sensitive=case_sensitive,
                    strict_middle_name=True,
                )
            )

            detailed_results = [result for result in search_results if isinstance(result, tuple)]
            if detailed_results:
                merged_frames = [result[0] for result in detailed_results]
                merged_df = pd.concat(merged_frames, ignore_index=True)
                merged_df = (
                    merged_df.groupby(["NPI", "Physician_Name", "Specialty"], as_index=False)
                    .agg(
                        Entry_Count=("Entry_Count", "max"),
                        Total_Payment_USD=("Total_Payment_USD", "max"),
                    )
                    .sort_values("Total_Payment_USD", ascending=False)
                )
                total_payment = float(merged_df["Total_Payment_USD"].sum())
                total_entries = int(merged_df["Entry_Count"].sum())
                results_by_year[year] = YearProcessingResult(
                    dataframe=merged_df.reset_index(drop=True),
                    total_payment=total_payment,
                    total_entries=total_entries,
                )
            else:
                total_payment = float(sum(result for result in search_results if isinstance(result, (int, float))))
                results_by_year[year] = YearProcessingResult(
                    dataframe=None,
                    total_payment=total_payment,
                    total_entries=0,
                )
        except Exception as exc:
            logger.error("Error processing %s data: %s", year, exc)
            results_by_year[year] = YearProcessingResult(dataframe=None, total_payment=0.0, total_entries=0)

    merged_result = merge_year_results(results_by_year)
    return merged_result.dataframe, merged_result.total_payment, merged_result.total_entries


def safe_filename(value: str) -> str:
    return re.sub(r"[^\w\s-]", "", value).strip().replace(" ", "_")


def year_range_label(years: Sequence[int]) -> str:
    return f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])


def format_currency_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    export_df = df.copy()
    for column in columns:
        export_df[column] = pd.to_numeric(export_df[column], errors="coerce").fillna(0.0)
        export_df[column] = export_df[column].map(lambda value: f"{value:.2f}")
    return export_df


def build_multi_npi_index(
    physician_data_by_name: Dict[str, Optional[pd.DataFrame]]
) -> Dict[str, List[Dict[str, object]]]:
    physicians_with_multiple_npis: Dict[str, List[Dict[str, object]]] = {}

    for name, df in physician_data_by_name.items():
        if df is None or df.empty or df["NPI"].nunique() <= 1:
            continue

        npi_rows = []
        for _, row in df.iterrows():
            npi_rows.append(
                {
                    "NPI": row["NPI"],
                    "Specialty": row["Specialty"],
                    "Entry_Count": int(row["Entry_Count"]),
                    "Total_USD": float(row["Total_USD"]),
                }
            )
        physicians_with_multiple_npis[name] = npi_rows

    return physicians_with_multiple_npis


def generate_html_dashboard(
    summary_df: pd.DataFrame,
    physicians_with_multiple_npis: Dict[str, List[Dict[str, object]]],
    years_to_process: Sequence[int],
    output_dir: Optional[str] = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"dashboard_{timestamp}.html" if output_dir else Path(f"dashboard_{timestamp}.html")

    year_columns = [f"Payment_{year}_USD" for year in years_to_process]
    summary = summary_df.copy()
    for column in [*year_columns, "Total_Payment"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)

    year_totals = {year: float(summary[f"Payment_{year}_USD"].sum()) for year in years_to_process}
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_html = []
    for _, row in summary.iterrows():
        name = " ".join(
            part for part in [row["First_Name"], row["Middle_Name"], row["Last_Name"]] if str(part).strip()
        )
        is_alert = name in physicians_with_multiple_npis
        alert_class = "alert" if is_alert else ""
        action = (
            f"<button class='button' onclick=\"toggleDetails('details_{safe_filename(name)}')\">View NPIs</button>"
            if is_alert
            else "-"
        )
        cells = [f"<td>{name}</td>"]
        cells.extend(f"<td>${row[f'Payment_{year}_USD']:,.2f}</td>" for year in years_to_process)
        cells.append(f"<td>${row['Total_Payment']:,.2f}</td>")
        cells.append(f"<td>{int(row['Total_Entries'])}</td>")
        cells.append(f"<td>{action}</td>")
        rows_html.append(f"<tr class='{alert_class}'>{''.join(cells)}</tr>")

    details_html = []
    if physicians_with_multiple_npis:
        details_html.append("<p>The following physicians have multiple NPIs, which may indicate source-data inconsistencies.</p><ul>")
        for name, npis in physicians_with_multiple_npis.items():
            details_html.append(
                f"<li><strong>{name}</strong>: {len(npis)} NPIs "
                f"<button class='button' onclick=\"toggleDetails('details_{safe_filename(name)}')\">View Details</button>"
                f"<div id='details_{safe_filename(name)}' class='details'>"
                "<table><tr><th>NPI</th><th>Specialty</th><th>Entries</th><th>Total Payment</th></tr>"
            )
            for npi_data in npis:
                details_html.append(
                    f"<tr><td>{npi_data['NPI']}</td><td>{npi_data['Specialty']}</td>"
                    f"<td>{npi_data['Entry_Count']}</td><td>${float(npi_data['Total_USD']):,.2f}</td></tr>"
                )
            details_html.append("</table></div></li>")
        details_html.append("</ul>")
    else:
        details_html.append("<p>No physicians with multiple NPIs found.</p>")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Research Payments Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .alert {{ background-color: #ffdddd; }}
        .details {{ display: none; padding: 10px; background-color: #f8f8f8; border: 1px solid #ddd; margin: 10px 0; }}
        .button {{ background-color: #4caf50; border: none; color: white; padding: 5px 10px; cursor: pointer; border-radius: 3px; }}
        .chart-container {{ height: 400px; margin-bottom: 30px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        function toggleDetails(id) {{
            const details = document.getElementById(id);
            details.style.display = details.style.display === "block" ? "none" : "block";
        }}
        function searchTable() {{
            const filter = document.getElementById("searchInput").value.toUpperCase();
            const rows = document.getElementById("summaryTable").getElementsByTagName("tr");
            for (let i = 0; i < rows.length; i++) {{
                const firstCell = rows[i].getElementsByTagName("td")[0];
                if (!firstCell) {{
                    continue;
                }}
                const matches = (firstCell.textContent || firstCell.innerText).toUpperCase().indexOf(filter) > -1;
                rows[i].style.display = matches ? "" : "none";
            }}
        }}
    </script>
</head>
<body>
    <h1>Research Payments Dashboard</h1>
    <p>Generated on {generated_at}</p>

    <h2>Summary Statistics</h2>
    <div class="chart-container">
        <canvas id="paymentsByYearChart"></canvas>
    </div>

    <h2>Physicians with Multiple NPIs</h2>
    {''.join(details_html)}

    <h2>Physician Payment Summary</h2>
    <input type="text" id="searchInput" onkeyup="searchTable()" placeholder="Search for names...">
    <table id="summaryTable">
        <thead>
            <tr>
                <th>Physician Name</th>
                {''.join(f'<th>Payment {year} ($)</th>' for year in years_to_process)}
                <th>Total Payment ($)</th>
                <th>Total Entries</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>

    <script>
        const yearTotals = {json.dumps(year_totals)};
        new Chart(document.getElementById('paymentsByYearChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(yearTotals),
                datasets: [{{
                    label: 'Total Payments by Year',
                    data: Object.values(yearTotals),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return '$' + context.parsed.y.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)

    return str(output_path)


def display_console_dashboard(
    summary_df: pd.DataFrame,
    physicians_with_multiple_npis: Dict[str, List[Dict[str, object]]],
    years_to_process: Sequence[int],
    physician_data_by_name: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
) -> None:
    summary = summary_df.copy()
    for column in [* [f"Payment_{year}_USD" for year in years_to_process], "Total_Payment"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)

    print("\n" + "=" * 80)
    print(f"{'RESEARCH PAYMENTS DASHBOARD':^80}")
    print("=" * 80)
    print("\nSUMMARY STATISTICS")
    print("-" * 80)

    total_physicians = len(summary)
    total_npis = 0
    if physician_data_by_name:
        total_npis = sum(df["NPI"].nunique() for df in physician_data_by_name.values() if df is not None and not df.empty)

    print(f"Total Physicians Processed: {total_physicians}")
    print(f"Total Unique NPIs: {total_npis}")
    print(f"Physicians with Multiple NPIs: {len(physicians_with_multiple_npis)}")

    print("\nPAYMENTS BY YEAR")
    print("-" * 80)
    for year in years_to_process:
        print(f"{year}: ${summary[f'Payment_{year}_USD'].sum():,.2f}")
    print(f"TOTAL: ${summary['Total_Payment'].sum():,.2f}")

    if physicians_with_multiple_npis:
        print("\nALERT: PHYSICIANS WITH MULTIPLE NPIs")
        print("-" * 80)
        for name, npis in physicians_with_multiple_npis.items():
            print(f"* {name}: {len(npis)} NPIs")
            for npi_data in npis:
                print(
                    f"  - NPI: {npi_data['NPI']}, Specialty: {npi_data['Specialty']}, "
                    f"Entries: {npi_data['Entry_Count']}, Total: ${float(npi_data['Total_USD']):,.2f}"
                )
            print()

    print("\nTOP 10 PHYSICIANS BY TOTAL PAYMENT")
    print("-" * 80)
    top10 = summary.nlargest(10, "Total_Payment")

    headers = ["Physician Name", *[f"{year} ($)" for year in years_to_process], "Total ($)", "Entries"]
    rows = []
    for _, row in top10.iterrows():
        name = " ".join(part for part in [row["First_Name"], row["Middle_Name"], row["Last_Name"]] if str(part).strip())
        rows.append(
            [
                name,
                *[f"${row[f'Payment_{year}_USD']:,.2f}" for year in years_to_process],
                f"${row['Total_Payment']:,.2f}",
                int(row["Total_Entries"]),
            ]
        )

    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("\n" + "=" * 80)


def process_physician_list(
    names_file: str,
    years_to_process: Sequence[int],
    output_dir: Optional[str] = None,
    case_sensitive: bool = False,
) -> None:
    physician_data_by_name: Dict[str, Optional[pd.DataFrame]] = {}
    names = read_names_from_file(names_file)
    if not names:
        logger.error("No names found in the input file.")
        return

    output_path = Path(output_dir) if output_dir else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    years_label = year_range_label(years_to_process)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = output_path / f"payment_summary_{years_label}_{timestamp}.csv"

    for index, raw_name in enumerate(names, start=1):
        logger.info("Processing physician %s/%s: %s", index, len(names), raw_name)
        first_name, middle_name, last_name = parse_name(raw_name)
        if not first_name or not last_name:
            logger.warning("Could not parse name properly: %s", raw_name)
            continue

        try:
            combined_df, total_payment, total_entries = process_physician(
                first_name, middle_name, last_name, years_to_process, case_sensitive
            )
            full_name = " ".join(part for part in [first_name, middle_name or "", last_name] if part).strip()
            physician_data_by_name[full_name] = combined_df

            summary_row = {
                "Full_Name": raw_name,
                "First_Name": first_name,
                "Middle_Name": middle_name or "",
                "Last_Name": last_name,
                "Total_Payment": total_payment,
                "Total_Entries": total_entries,
            }
            for year in years_to_process:
                column = f"Payment_{year}_USD"
                summary_row[column] = float(combined_df[column].sum()) if combined_df is not None and column in combined_df.columns else 0.0
            summary_rows.append(summary_row)

            if combined_df is not None and not combined_df.empty:
                detail_filename = output_path / f"research_payments_{safe_filename(raw_name)}_{years_label}.csv"
                export_df = format_currency_columns(
                    combined_df,
                    [*[f"Payment_{year}_USD" for year in years_to_process], "Total_USD"],
                )
                export_df.to_csv(detail_filename, index=False)
                print(f"\nResults for {raw_name}:")
                print(f"Total Payment: ${total_payment:,.2f}")
                print(f"Total Entries: {total_entries}")
                print(f"Saved to: {detail_filename}")
            else:
                print(f"\nNo detailed results for {raw_name}. Total payment: ${total_payment:,.2f}")
        except Exception as exc:
            logger.error("Error processing physician %s: %s", raw_name, exc)
            summary_row = {
                "Full_Name": raw_name,
                "First_Name": first_name,
                "Middle_Name": middle_name or "",
                "Last_Name": last_name,
                "Total_Payment": 0.0,
                "Total_Entries": 0,
                "Error": str(exc),
            }
            for year in years_to_process:
                summary_row[f"Payment_{year}_USD"] = 0.0
            summary_rows.append(summary_row)

    if not summary_rows:
        logger.warning("No summary data to save.")
        return

    summary_df = pd.DataFrame(summary_rows)
    ordered_columns = [
        "Full_Name",
        "First_Name",
        "Middle_Name",
        "Last_Name",
        *[f"Payment_{year}_USD" for year in sorted(years_to_process)],
        "Total_Payment",
        "Total_Entries",
    ]
    if "Error" in summary_df.columns:
        ordered_columns.append("Error")

    summary_df = summary_df[ordered_columns].sort_values("Total_Payment", ascending=False).reset_index(drop=True)
    summary_export = format_currency_columns(
        summary_df,
        [*[f"Payment_{year}_USD" for year in years_to_process], "Total_Payment"],
    )
    summary_export.to_csv(summary_filename, index=False)
    logger.info("Saved summary results to: %s", summary_filename)

    physicians_with_multiple_npis = build_multi_npi_index(physician_data_by_name)
    display_console_dashboard(summary_df, physicians_with_multiple_npis, years_to_process, physician_data_by_name)
    dashboard_file = generate_html_dashboard(summary_df, physicians_with_multiple_npis, years_to_process, str(output_path))

    print(f"\nInteractive dashboard saved to: {dashboard_file}")
    print("Open this file in a web browser to view the interactive dashboard.")

    try:
        open_now = input("Open dashboard now? (y/n): ").strip().lower()
        if open_now == "y":
            webbrowser.open("file://" + os.path.abspath(dashboard_file))
    except Exception as exc:
        logger.error("Error opening dashboard: %s", exc)


def save_single_physician_results(
    combined_df: Optional[pd.DataFrame],
    total_payment: float,
    total_entries: int,
    first_name: str,
    middle_name: Optional[str],
    last_name: str,
    years: Sequence[int],
    output_dir: Optional[str],
) -> None:
    display_name = " ".join(part for part in [first_name, middle_name or "", last_name] if part).strip()
    print(f"\nResearch payments for {display_name}:")
    print("-" * 80)
    print(f"Total Payment: ${total_payment:,.2f}")
    print(f"Total Entries: {total_entries}")

    if combined_df is None or combined_df.empty:
        print(f"\nNo detailed results. Total payment: ${total_payment:,.2f}")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", "${:,.2f}".format)

    print("\nDetailed Results:")
    print("-" * 80)
    print(combined_df.to_string(index=False))

    output_path = Path(output_dir) if output_dir else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"research_payments_{last_name}_{first_name}_{year_range_label(years)}.csv"

    export_df = format_currency_columns(
        combined_df,
        [*[f"Payment_{year}_USD" for year in years], "Total_USD"],
    )
    export_df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract research payments for physicians from CMS Open Payments data."
    )
    parser.add_argument("--names_file", type=str, help="Path to file containing physician names")
    parser.add_argument("--years", type=int, nargs="+", default=[2015, 2016], help="Years to process")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--case_sensitive", action="store_true", help="Use case-sensitive name matching")
    parser.add_argument("--first_name", type=str, help="Physician first name (for single mode)")
    parser.add_argument("--middle_name", type=str, help="Physician middle name (for single mode)")
    parser.add_argument("--last_name", type=str, help="Physician last name (for single mode)")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.names_file:
        logger.info("Running in batch mode with names from: %s", args.names_file)
        process_physician_list(args.names_file, args.years, args.output_dir, args.case_sensitive)
        return

    if args.first_name and args.last_name:
        logger.info(
            "Running in single physician mode for: %s %s %s",
            args.first_name,
            args.middle_name or "",
            args.last_name,
        )
        combined_df, total_payment, total_entries = process_physician(
            args.first_name,
            args.middle_name,
            args.last_name,
            args.years,
            args.case_sensitive,
        )
        save_single_physician_results(
            combined_df,
            total_payment,
            total_entries,
            args.first_name,
            args.middle_name,
            args.last_name,
            args.years,
            args.output_dir,
        )
        return

    print("No arguments provided. Running with example values:")
    example_first = "Benjamin"
    example_middle = "G"
    example_last = "Domb"
    example_years = [2015, 2016]
    print(f"Physician: {example_first} {example_middle} {example_last}")
    print(f"Years: {example_years}")

    combined_df, total_payment, total_entries = process_physician(
        example_first,
        example_middle,
        example_last,
        example_years,
        False,
    )
    save_single_physician_results(
        combined_df,
        total_payment,
        total_entries,
        example_first,
        example_middle,
        example_last,
        example_years,
        None,
    )


if __name__ == "__main__":
    main()
