from pathlib import Path

import pandas as pd

from research_payments.cli import main
from tests.helpers import covered_row, write_open_payments_csv


def test_cli_single_physician_creates_detail_csv(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    write_open_payments_csv(
        tmp_path / "2015_rsh_payments.csv",
        [
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="1154454635",
                amount=200.0,
                specialty="Orthopedic Surgery",
            )
        ],
    )
    output_dir = tmp_path / "results"
    monkeypatch.chdir(tmp_path)

    main(
        [
            "--first_name",
            "Benjamin",
            "--middle_name",
            "G",
            "--last_name",
            "Domb",
            "--years",
            "2015",
            "--output_dir",
            str(output_dir),
        ]
    )

    output = capsys.readouterr().out
    assert "Total Payment: $200.00" in output

    detail_csv = output_dir / "research_payments_Domb_Benjamin_2015.csv"
    assert detail_csv.exists()
    detail_df = pd.read_csv(detail_csv)
    assert str(detail_df.loc[0, "NPI"]) == "1154454635"
    assert detail_df.loc[0, "Physician_Name"] == "Benjamin G Domb"


def test_cli_batch_mode_creates_summary_dashboard_and_detail_files(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    write_open_payments_csv(
        tmp_path / "2015_rsh_payments.csv",
        [
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="1154454635",
                amount=200.0,
                specialty="Orthopedic Surgery",
            ),
            covered_row(
                first_name="Alice",
                middle_name="M",
                last_name="Smith",
                npi="2233445566",
                amount=90.0,
                specialty="Cardiology",
            ),
        ],
    )
    names_file = tmp_path / "physicians.txt"
    names_file.write_text("Benjamin G Domb\nAlice M Smith\n", encoding="utf-8")
    output_dir = tmp_path / "results"
    monkeypatch.chdir(tmp_path)

    main(
        [
            "--names_file",
            str(names_file),
            "--years",
            "2015",
            "--output_dir",
            str(output_dir),
        ]
    )

    output = capsys.readouterr().out
    assert "Interactive dashboard saved to:" in output

    summary_files = list(output_dir.glob("payment_summary_2015_*.csv"))
    dashboard_files = list(output_dir.glob("dashboard_*.html"))
    detail_files = list(output_dir.glob("research_payments_*_2015.csv"))

    assert len(summary_files) == 1
    assert len(dashboard_files) == 1
    assert len(detail_files) == 2

    summary_df = pd.read_csv(summary_files[0])
    assert set(summary_df["Full_Name"]) == {"Benjamin G Domb", "Alice M Smith"}
    assert set(summary_df["Total_Payment"]) == {200.0, 90.0}
