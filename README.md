# Research Payments Extractor

Build a searchable dashboard from a ZIP archive of CMS Open Payments research-payment CSVs, or run direct physician queries from the command line.

## Project Layout

The code is organized in a standard `src/` layout:

- `research_extractor.py`: compatibility entrypoint
- `run_physician_query.sh`: user-facing shell launcher
- `src/research_payments/cli.py`: command-line flow
- `src/research_payments/dataset.py`: CSV loading and matching
- `src/research_payments/processor.py`: multi-year aggregation
- `src/research_payments/reporting.py`: CSV, console, and HTML output

## Requirements

- Python 3.10+
- Python packages:
  - `pandas`
  - `tabulate`

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you prefer not to activate the environment, run commands with:

```bash
.venv/bin/python research_extractor.py --help
```

Or use the shell launcher:

```bash
./run_physician_query.sh --help
```

## Running Tests

Install dependencies in the local virtual environment, then run:

```bash
.venv/bin/python -m pytest -q
```

The test suite includes:

- unit tests for name matching
- unit tests for dataset search and identity extraction
- unit tests for multi-year aggregation
- integration tests for single-physician CLI runs
- integration tests for batch CLI runs
- an entrypoint integration test for `research_extractor.py`

## Input Files

Download CMS Open Payments research-payment CSVs and place them in the project directory, named exactly like this:

```text
2015_rsh_payments.csv
2016_rsh_payments.csv
2017_rsh_payments.csv
```

The year in the filename must match the year you pass on the command line.

## Primary Workflow

1. Put the dated research-payment CSV files into a single ZIP archive.
2. Run the shell launcher with `--zip_file`.
3. Open the generated HTML dashboard.
4. Type a physician name into the search box.
5. Review every indexed appearance of that name, including rows with and without a unique identifier.

Example:

```bash
./run_physician_query.sh \
  --zip_file research_files.zip \
  --dashboard_output results/research_search_dashboard.html
```

The search dashboard shows:

- physician name
- role in the record
- year
- source file
- total payment for that grouped appearance
- number of instances
- unique identifier when present
- explicit missing-identifier state when not present

## Usage

Run from the repository root.

### ZIP Archive Search Dashboard

Preferred command:

```bash
./run_physician_query.sh \
  --zip_file research_files.zip \
  --dashboard_output results/research_search_dashboard.html
```

### Single Physician

Preferred command:

```bash
./run_physician_query.sh \
  --first_name Benjamin \
  --middle_name G \
  --last_name Domb \
  --years 2015 2016
```

Direct Python command:

```bash
python3 research_extractor.py \
  --first_name Benjamin \
  --middle_name G \
  --last_name Domb \
  --years 2015 2016
```

Optional flags:

- `--output_dir results`
- `--case_sensitive`

Example:

```bash
python3 research_extractor.py \
  --first_name Jane \
  --last_name Smith \
  --years 2019 2020 2021 \
  --output_dir results
```

### Batch Mode

Create a text file with one physician name per line:

```text
Benjamin G Domb
Jane Smith
John A Doe
```

Run:

```bash
./run_physician_query.sh \
  --names_file physicians.txt \
  --years 2019 2020 2021 \
  --output_dir results
```

## Outputs

### Single Physician Output

The script prints:

- Total payment amount
- Total matched entries
- Detailed per-NPI breakdown

It also writes a CSV like:

```text
research_payments_Domb_Benjamin_2015-2016.csv
```

### Batch Output

The script generates:

- One detail CSV per physician with matches
- One summary CSV for the whole run
- One HTML dashboard
- A console dashboard

Typical output files:

```text
results/payment_summary_2019-2021_YYYYMMDD_HHMMSS.csv
results/research_payments_Benjamin_G_Domb_2019-2021.csv
results/dashboard_YYYYMMDD_HHMMSS.html
```

## Name Matching Behavior

- Matches covered-recipient names and principal-investigator names
- Supports optional middle-name input
- If a middle name is provided, the tool checks both:
  - with the middle name
  - without the middle name
- Supports middle initials such as `G` or `G.`
- `--case_sensitive` switches matching from normalized matching to exact case matching

## Notes

- The script expects CMS research-payment CSV schemas used in older `Physician_*` files and newer `Covered_Recipient_*` files.
- If a physician has multiple NPIs, the dashboard highlights that case.
- If no CLI arguments are provided, the script runs a built-in example query.

## Troubleshooting

### `CSV file not found`

Make sure the file exists in the repository root and matches the expected naming format:

```text
<year>_rsh_payments.csv
```

### No results returned

Check:

- spelling of the physician name
- year selection
- whether the physician appears under a different middle-name format
- whether the relevant CMS file is the research-payments dataset

### Import or dependency errors

Install dependencies again:

```bash
python3 -m pip install -r requirements.txt
```
