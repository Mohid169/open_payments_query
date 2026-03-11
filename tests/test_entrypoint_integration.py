import os
import subprocess
import sys
from pathlib import Path

from tests.helpers import covered_row, write_open_payments_csv


def test_python_entrypoint_runs_from_repo_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    write_open_payments_csv(
        tmp_path / "2015_rsh_payments.csv",
        [
            covered_row(
                first_name="Benjamin",
                middle_name="G",
                last_name="Domb",
                npi="1154454635",
                amount=220.0,
                specialty="Orthopedic Surgery",
            )
        ],
    )

    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + existing_path if existing_path else "")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "research_extractor.py"),
            "--first_name",
            "Benjamin",
            "--middle_name",
            "G",
            "--last_name",
            "Domb",
            "--years",
            "2015",
            "--output_dir",
            str(tmp_path / "results"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Total Payment: $220.00" in result.stdout
