from pathlib import Path

import pandas as pd


def write_open_payments_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def covered_row(
    *,
    first_name: str,
    last_name: str,
    amount: float,
    middle_name: str = "",
    npi: str = "",
    profile_id: str = "",
    specialty: str = "",
) -> dict:
    return {
        "Covered_Recipient_First_Name": first_name,
        "Covered_Recipient_Middle_Name": middle_name,
        "Covered_Recipient_Last_Name": last_name,
        "Covered_Recipient_NPI": npi,
        "Covered_Recipient_Profile_ID": profile_id,
        "Covered_Recipient_Specialty_1": specialty,
        "Principal_Investigator_1_First_Name": "",
        "Principal_Investigator_1_Middle_Name": "",
        "Principal_Investigator_1_Last_Name": "",
        "Principal_Investigator_1_NPI": "",
        "Total_Amount_of_Payment_USDollars": amount,
    }


def pi_row(
    *,
    covered_first_name: str,
    covered_last_name: str,
    pi_first_name: str,
    pi_last_name: str,
    amount: float,
    pi_middle_name: str = "",
    pi_npi: str = "",
) -> dict:
    return {
        "Covered_Recipient_First_Name": covered_first_name,
        "Covered_Recipient_Middle_Name": "",
        "Covered_Recipient_Last_Name": covered_last_name,
        "Covered_Recipient_NPI": "",
        "Covered_Recipient_Profile_ID": "",
        "Covered_Recipient_Specialty_1": "",
        "Principal_Investigator_1_First_Name": pi_first_name,
        "Principal_Investigator_1_Middle_Name": pi_middle_name,
        "Principal_Investigator_1_Last_Name": pi_last_name,
        "Principal_Investigator_1_NPI": pi_npi,
        "Total_Amount_of_Payment_USDollars": amount,
    }
