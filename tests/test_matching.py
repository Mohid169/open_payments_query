import pandas as pd

from research_payments.matching import middle_name_mask, name_match_mask
from research_payments.models import PhysicianQuery


def test_middle_name_mask_supports_initial_matching() -> None:
    series = pd.Series(["G", "G.", "Grant", "X"])
    mask = middle_name_mask(series, "G.", case_sensitive=False)
    assert mask.tolist() == [True, True, True, False]


def test_name_match_mask_is_case_insensitive_by_default() -> None:
    df = pd.DataFrame(
        {
            "first": ["Benjamin", "Alice"],
            "middle": ["G", ""],
            "last": ["Domb", "Smith"],
        }
    )
    query = PhysicianQuery(first_name="benjamin", middle_name="g", last_name="domb")
    mask = name_match_mask(df, query, "first", "last", "middle")
    assert mask.tolist() == [True, False]


def test_name_match_mask_respects_case_sensitive_flag() -> None:
    df = pd.DataFrame(
        {
            "first": ["Benjamin", "benjamin"],
            "middle": ["G", "G"],
            "last": ["Domb", "Domb"],
        }
    )
    query = PhysicianQuery(
        first_name="Benjamin",
        middle_name="G",
        last_name="Domb",
        case_sensitive=True,
    )
    mask = name_match_mask(df, query, "first", "last", "middle")
    assert mask.tolist() == [True, False]
