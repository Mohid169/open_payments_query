from pathlib import Path

from research_payments.dataset import OpenPaymentsDataset
from research_payments.models import PhysicianQuery
from tests.helpers import covered_row, pi_row, write_open_payments_csv


def test_dataset_search_matches_covered_recipient_and_sums_payments(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "2015_rsh_payments.csv"
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
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="1154454635",
                amount=150.0,
                specialty="Orthopedic Surgery",
            ),
        ],
    )

    dataset = OpenPaymentsDataset(csv_path)
    result = dataset.search(PhysicianQuery("Benjamin", "Domb", "G"))

    assert len(result.dataframe) == 1
    row = result.dataframe.iloc[0]
    assert row["NPI"] == "1154454635"
    assert row["Physician_Name"] == "Benjamin G Domb"
    assert row["Specialty"] == "Orthopedic Surgery"
    assert row["Entry_Count"] == 2
    assert row["Total_Payment_USD"] == 250.0


def test_dataset_search_matches_principal_investigator_when_covered_name_differs(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "2016_rsh_payments.csv"
    write_open_payments_csv(
        csv_path,
        [
            pi_row(
                covered_first_name="Study",
                covered_last_name="Subject",
                pi_first_name="Benjamin",
                pi_middle_name="G",
                pi_last_name="Domb",
                pi_npi="1154454635",
                amount=400.0,
            )
        ],
    )

    dataset = OpenPaymentsDataset(csv_path)
    result = dataset.search(PhysicianQuery("Benjamin", "Domb", "G"))

    assert len(result.dataframe) == 1
    row = result.dataframe.iloc[0]
    assert row["NPI"] == "1154454635"
    assert row["Physician_Name"] == "Benjamin G Domb"
    assert row["Specialty"] == "Principal Investigator"
    assert row["Total_Payment_USD"] == 400.0


def test_dataset_does_not_stringify_nan_into_identity_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "2017_rsh_payments.csv"
    write_open_payments_csv(
        csv_path,
        [
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="",
                profile_id="12345",
                amount=50.0,
            )
        ],
    )

    dataset = OpenPaymentsDataset(csv_path)
    result = dataset.search(PhysicianQuery("Benjamin", "Domb", "G"))

    row = result.dataframe.iloc[0]
    assert row["NPI"] == "PROFILE_12345"
    assert row["Physician_Name"] == "Benjamin G Domb"
    assert row["Specialty"] == "Unknown"
