import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from research_payments.processor import ResearchPaymentsProcessor
from research_payments.reporting import (
    collect_multi_npi_physicians,
    display_console_dashboard,
    generate_html_dashboard,
    maybe_open_dashboard,
    save_detail_csv,
    save_single_physician_results,
)
from research_payments.utils import format_currency_columns, parse_name, read_names_from_file, safe_filename, year_range_label


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract research payments for physicians from CMS Open Payments data."
    )
    parser.add_argument("--names_file", type=str, help="Path to file containing physician names")
    parser.add_argument("--years", type=int, nargs="+", default=[2015, 2016], help="Years to process")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--case_sensitive", action="store_true", help="Use case-sensitive name matching")
    parser.add_argument("--first_name", type=str, help="Physician first name")
    parser.add_argument("--middle_name", type=str, help="Physician middle name")
    parser.add_argument("--last_name", type=str, help="Physician last name")
    return parser


def process_physician_list(
    names_file: str,
    years_to_process: Sequence[int],
    output_dir: Optional[str] = None,
    case_sensitive: bool = False,
) -> None:
    output_root = Path(output_dir) if output_dir else Path(".")
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        names = read_names_from_file(names_file)
    except OSError as exc:
        logger.error("Error reading names file %s: %s", names_file, exc)
        return

    if not names:
        logger.error("No names found in %s", names_file)
        return

    processor = ResearchPaymentsProcessor()
    physician_data_by_name: dict[str, pd.DataFrame | None] = {}
    summary_rows = []
    years_label = year_range_label(years_to_process)
    summary_path = output_root / f"payment_summary_{years_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    for index, raw_name in enumerate(names, start=1):
        logger.info("Processing physician %s/%s: %s", index, len(names), raw_name)
        first_name, middle_name, last_name = parse_name(raw_name)
        if not first_name or not last_name:
            logger.warning("Skipping unparsable name: %s", raw_name)
            continue

        try:
            result = processor.process_physician(
                first_name=first_name,
                middle_name=middle_name,
                last_name=last_name,
                years_to_process=years_to_process,
                case_sensitive=case_sensitive,
            )
            display_name = " ".join(part for part in [first_name, middle_name or "", last_name] if part).strip()
            physician_data_by_name[display_name] = result.dataframe

            row = {
                "Full_Name": raw_name,
                "First_Name": first_name,
                "Middle_Name": middle_name or "",
                "Last_Name": last_name,
                "Total_Payment": result.total_payment,
                "Total_Entries": result.total_entries,
            }
            for year in years_to_process:
                column = f"Payment_{year}_USD"
                row[column] = (
                    float(result.dataframe[column].sum())
                    if result.dataframe is not None and column in result.dataframe.columns
                    else 0.0
                )
            summary_rows.append(row)

            if result.dataframe is not None and not result.dataframe.empty:
                destination = output_root / f"research_payments_{safe_filename(raw_name)}_{years_label}.csv"
                save_detail_csv(result.dataframe, years_to_process, destination)
                print(f"\nResults for {raw_name}:")
                print(f"Total Payment: ${result.total_payment:,.2f}")
                print(f"Total Entries: {result.total_entries}")
                print(f"Saved to: {destination}")
            else:
                print(f"\nNo detailed results for {raw_name}. Total payment: ${result.total_payment:,.2f}")
        except Exception as exc:
            logger.error("Error processing physician %s: %s", raw_name, exc)
            error_row = {
                "Full_Name": raw_name,
                "First_Name": first_name,
                "Middle_Name": middle_name or "",
                "Last_Name": last_name,
                "Total_Payment": 0.0,
                "Total_Entries": 0,
                "Error": str(exc),
            }
            for year in years_to_process:
                error_row[f"Payment_{year}_USD"] = 0.0
            summary_rows.append(error_row)

    if not summary_rows:
        logger.warning("No summary data generated")
        return

    summary_df = pd.DataFrame(summary_rows)
    ordered_columns = [
        "Full_Name",
        "First_Name",
        "Middle_Name",
        "Last_Name",
        *[f"Payment_{year}_USD" for year in years_to_process],
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
    summary_export.to_csv(summary_path, index=False)
    logger.info("Saved summary CSV to %s", summary_path)

    physicians_with_multiple_npis = collect_multi_npi_physicians(physician_data_by_name)
    display_console_dashboard(summary_df, physicians_with_multiple_npis, years_to_process, physician_data_by_name)
    dashboard_path = generate_html_dashboard(summary_df, physicians_with_multiple_npis, years_to_process, str(output_root))
    print(f"\nInteractive dashboard saved to: {dashboard_path}")
    maybe_open_dashboard(dashboard_path)


def main() -> None:
    args = build_parser().parse_args()
    processor = ResearchPaymentsProcessor()

    if args.names_file:
        process_physician_list(args.names_file, args.years, args.output_dir, args.case_sensitive)
        return

    if args.first_name and args.last_name:
        result = processor.process_physician(
            first_name=args.first_name,
            middle_name=args.middle_name,
            last_name=args.last_name,
            years_to_process=args.years,
            case_sensitive=args.case_sensitive,
        )
        save_single_physician_results(
            result=result,
            first_name=args.first_name,
            middle_name=args.middle_name,
            last_name=args.last_name,
            years_to_process=args.years,
            output_dir=args.output_dir,
        )
        return

    print("No arguments provided. Running example query.")
    result = processor.process_physician(
        first_name="Benjamin",
        middle_name="G",
        last_name="Domb",
        years_to_process=[2015, 2016],
        case_sensitive=False,
    )
    save_single_physician_results(
        result=result,
        first_name="Benjamin",
        middle_name="G",
        last_name="Domb",
        years_to_process=[2015, 2016],
        output_dir=None,
    )
