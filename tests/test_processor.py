from pathlib import Path

from research_payments.processor import ResearchPaymentsProcessor
from tests.helpers import covered_row, write_open_payments_csv


def test_processor_combines_years_and_keeps_distinct_payment_columns(
    tmp_path: Path, monkeypatch
) -> None:
    write_open_payments_csv(
        tmp_path / "2015_rsh_payments.csv",
        [
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="1154454635",
                amount=100.0,
                specialty="Orthopedic Surgery",
            )
        ],
    )
    write_open_payments_csv(
        tmp_path / "2016_rsh_payments.csv",
        [
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="1154454635",
                amount=300.0,
                specialty="Orthopedic Surgery",
            )
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = ResearchPaymentsProcessor().process_physician(
        first_name="Benjamin",
        middle_name="G",
        last_name="Domb",
        years_to_process=[2015, 2016],
    )

    assert result.total_payment == 400.0
    assert result.total_entries == 2
    row = result.dataframe.iloc[0]
    assert row["Payment_2015_USD"] == 100.0
    assert row["Payment_2016_USD"] == 300.0
    assert row["Total_USD"] == 400.0


def test_processor_middle_name_dual_search_does_not_double_count_same_year(
    tmp_path: Path, monkeypatch
) -> None:
    write_open_payments_csv(
        tmp_path / "2015_rsh_payments.csv",
        [
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="1154454635",
                amount=125.0,
                specialty="Orthopedic Surgery",
            )
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = ResearchPaymentsProcessor().process_physician(
        first_name="Benjamin",
        middle_name="G",
        last_name="Domb",
        years_to_process=[2015],
    )

    assert result.total_payment == 125.0
    assert result.total_entries == 1
