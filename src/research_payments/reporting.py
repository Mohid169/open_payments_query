import html
import json
import logging
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tabulate import tabulate

from research_payments.models import PhysicianResult
from research_payments.utils import format_currency_columns, safe_filename, year_range_label


logger = logging.getLogger(__name__)


def collect_multi_npi_physicians(
    physician_data_by_name: Dict[str, Optional[pd.DataFrame]]
) -> Dict[str, List[Dict[str, object]]]:
    physicians = {}
    for name, df in physician_data_by_name.items():
        if df is None or df.empty or df["NPI"].nunique() <= 1:
            continue

        details = []
        for _, row in df.iterrows():
            details.append(
                {
                    "NPI": row["NPI"],
                    "Specialty": row["Specialty"],
                    "Entry_Count": int(row["Entry_Count"]),
                    "Total_USD": float(row["Total_USD"]),
                }
            )
        physicians[name] = details
    return physicians


def save_detail_csv(
    combined_df: pd.DataFrame,
    years_to_process: Sequence[int],
    destination: Path,
) -> None:
    payment_columns = [f"Payment_{year}_USD" for year in years_to_process]
    export_df = format_currency_columns(combined_df, [*payment_columns, "Total_USD"])
    export_df.to_csv(destination, index=False)


def save_single_physician_results(
    result: PhysicianResult,
    first_name: str,
    middle_name: Optional[str],
    last_name: str,
    years_to_process: Sequence[int],
    output_dir: Optional[str],
) -> None:
    display_name = " ".join(part for part in [first_name, middle_name or "", last_name] if part).strip()
    print(f"\nResearch payments for {display_name}:")
    print("-" * 80)
    print(f"Total Payment: ${result.total_payment:,.2f}")
    print(f"Total Entries: {result.total_entries}")

    if result.dataframe is None or result.dataframe.empty:
        print("\nNo detailed results found.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", "${:,.2f}".format)
    print("\nDetailed Results:")
    print("-" * 80)
    print(result.dataframe.to_string(index=False))

    output_root = Path(output_dir) if output_dir else Path(".")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"research_payments_{last_name}_{first_name}_{year_range_label(years_to_process)}.csv"
    save_detail_csv(result.dataframe, years_to_process, destination)
    print(f"\nResults saved to: {destination}")


def display_console_dashboard(
    summary_df: pd.DataFrame,
    physicians_with_multiple_npis: Dict[str, List[Dict[str, object]]],
    years_to_process: Sequence[int],
    physician_data_by_name: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
) -> None:
    summary = summary_df.copy()
    payment_columns = [f"Payment_{year}_USD" for year in years_to_process]
    for column in [*payment_columns, "Total_Payment"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)

    total_npis = 0
    if physician_data_by_name:
        total_npis = sum(
            df["NPI"].nunique()
            for df in physician_data_by_name.values()
            if df is not None and not df.empty
        )

    print("\n" + "=" * 80)
    print(f"{'RESEARCH PAYMENTS DASHBOARD':^80}")
    print("=" * 80)
    print("\nSUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total Physicians Processed: {len(summary)}")
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
        for name, details in physicians_with_multiple_npis.items():
            print(f"* {name}: {len(details)} NPIs")
            for detail in details:
                print(
                    f"  - NPI: {detail['NPI']}, Specialty: {detail['Specialty']}, "
                    f"Entries: {detail['Entry_Count']}, Total: ${float(detail['Total_USD']):,.2f}"
                )
            print()

    top10 = summary.nlargest(10, "Total_Payment")
    headers = ["Physician Name", *[f"{year} ($)" for year in years_to_process], "Total ($)", "Entries"]
    rows = []
    for _, row in top10.iterrows():
        display_name = " ".join(
            part for part in [row["First_Name"], row["Middle_Name"], row["Last_Name"]] if str(part).strip()
        )
        rows.append(
            [
                display_name,
                *[f"${row[f'Payment_{year}_USD']:,.2f}" for year in years_to_process],
                f"${row['Total_Payment']:,.2f}",
                int(row["Total_Entries"]),
            ]
        )

    print("\nTOP 10 PHYSICIANS BY TOTAL PAYMENT")
    print("-" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("\n" + "=" * 80)


def generate_html_dashboard(
    summary_df: pd.DataFrame,
    physicians_with_multiple_npis: Dict[str, List[Dict[str, object]]],
    years_to_process: Sequence[int],
    output_dir: Optional[str] = None,
) -> str:
    output_root = Path(output_dir) if output_dir else Path(".")
    output_root.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_root / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    summary = summary_df.copy()
    payment_columns = [f"Payment_{year}_USD" for year in years_to_process]
    for column in [*payment_columns, "Total_Payment"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)

    year_totals = {str(year): float(summary[f"Payment_{year}_USD"].sum()) for year in years_to_process}

    multi_npi_sections = []
    if physicians_with_multiple_npis:
        multi_npi_sections.append(
            "<p>The following physicians have multiple NPIs, which may indicate source-data inconsistencies.</p><ul>"
        )
        for name, details in physicians_with_multiple_npis.items():
            safe_id = safe_filename(name)
            multi_npi_sections.append(
                f"<li><strong>{html.escape(name)}</strong>: {len(details)} NPIs "
                f"<button class='button' onclick=\"toggleDetails('{safe_id}')\">View Details</button>"
                f"<div id='{safe_id}' class='details'>"
                "<table><tr><th>NPI</th><th>Specialty</th><th>Entries</th><th>Total Payment</th></tr>"
            )
            for detail in details:
                multi_npi_sections.append(
                    "<tr>"
                    f"<td>{html.escape(str(detail['NPI']))}</td>"
                    f"<td>{html.escape(str(detail['Specialty']))}</td>"
                    f"<td>{int(detail['Entry_Count'])}</td>"
                    f"<td>${float(detail['Total_USD']):,.2f}</td>"
                    "</tr>"
                )
            multi_npi_sections.append("</table></div></li>")
        multi_npi_sections.append("</ul>")
    else:
        multi_npi_sections.append("<p>No physicians with multiple NPIs found.</p>")

    table_rows = []
    for _, row in summary.iterrows():
        display_name = " ".join(
            part for part in [row["First_Name"], row["Middle_Name"], row["Last_Name"]] if str(part).strip()
        )
        is_alert = display_name in physicians_with_multiple_npis
        row_class = "alert" if is_alert else ""
        detail_button = (
            f"<button class='button' onclick=\"toggleDetails('{safe_filename(display_name)}')\">View NPIs</button>"
            if is_alert
            else "-"
        )

        cells = [f"<td>{html.escape(display_name)}</td>"]
        for year in years_to_process:
            cells.append(f"<td>${row[f'Payment_{year}_USD']:,.2f}</td>")
        cells.append(f"<td>${row['Total_Payment']:,.2f}</td>")
        cells.append(f"<td>{int(row['Total_Entries'])}</td>")
        cells.append(f"<td>{detail_button}</td>")
        table_rows.append(f"<tr class='{row_class}'>{''.join(cells)}</tr>")

    html_output = f"""<!DOCTYPE html>
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
        .button {{ background-color: #4caf50; border: none; color: white; padding: 5px 10px; border-radius: 3px; cursor: pointer; }}
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
                const text = firstCell.textContent || firstCell.innerText;
                rows[i].style.display = text.toUpperCase().includes(filter) ? "" : "none";
            }}
        }}
    </script>
</head>
<body>
    <h1>Research Payments Dashboard</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Summary Statistics</h2>
    <div class="chart-container">
        <canvas id="paymentsByYearChart"></canvas>
    </div>
    <h2>Physicians with Multiple NPIs</h2>
    {''.join(multi_npi_sections)}
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
            {''.join(table_rows)}
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

    with open(dashboard_path, "w", encoding="utf-8") as handle:
        handle.write(html_output)

    return str(dashboard_path)


def maybe_open_dashboard(dashboard_path: str) -> None:
    if not sys.stdin.isatty():
        return

    try:
        if input("Open dashboard now? (y/n): ").strip().lower() == "y":
            webbrowser.open("file://" + os.path.abspath(dashboard_path))
    except Exception as exc:
        logger.error("Error opening dashboard: %s", exc)
