from pathlib import Path
import zipfile

from research_payments.archive_index import build_searchable_records_from_zip
from tests.helpers import covered_row, pi_row, write_open_payments_csv


def test_build_searchable_records_from_zip_indexes_covered_and_pi_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "2021_rsh_payments.csv"
    write_open_payments_csv(
        csv_path,
        [
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="1154454635",
                amount=100.0,
                specialty="Orthopedic Surgery",
            ),
            pi_row(
                covered_first_name="Study",
                covered_last_name="Subject",
                pi_first_name="Benjamin",
                pi_middle_name="G",
                pi_last_name="Domb",
                pi_npi="1154454635",
                amount=80.0,
            ),
            covered_row(
                first_name="Benjamin",
                middle_name="",
                last_name="Domb",
                npi="",
                profile_id="999",
                amount=40.0,
                specialty="Orthopedic Surgery",
            ),
        ],
    )

    zip_path = tmp_path / "research_files.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="2021_rsh_payments.csv")

    records = build_searchable_records_from_zip(zip_path)
    assert len(records) == 4

    identifier_types = {(record["role"], record["identifier_type"]) for record in records}
    assert ("Covered Recipient", "NPI") in identifier_types
    assert ("Covered Recipient", "Profile ID") in identifier_types
    assert ("Principal Investigator 1", "NPI") in identifier_types
    assert ("Covered Recipient", "Missing") in identifier_types


def test_build_searchable_records_from_zip_extracts_year_from_member_name(tmp_path: Path) -> None:
    csv_path = tmp_path / "payments.csv"
    write_open_payments_csv(
        csv_path,
        [covered_row(first_name="Alice", last_name="Smith", amount=50.0, npi="123")],
    )

    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="nested/2023_rsh_payments.csv")

    records = build_searchable_records_from_zip(zip_path)
    assert records[0]["year"] == "2023"
