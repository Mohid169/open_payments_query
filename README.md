# Research Payments Extractor

Extract and summarize CMS Open Payments research-payment data for one physician or a batch of physicians across multiple years.

## Project Layout

The repository now uses a small package instead of one large script:

- `research_extractor.py`: thin entrypoint
- `research_payments/cli.py`: command-line flow
- `research_payments/dataset.py`: CSV loading and matching
- `research_payments/processor.py`: multi-year aggregation
- `research_payments/reporting.py`: CSV, console, and HTML output

## Requirements

- Python 3.10+
- Python packages:
  - `pandas`
  - `tabulate`

Install dependencies with:

```bash
python3 -m pip install pandas tabulate
```

## Input Files

Download CMS Open Payments research-payment CSVs and place them in the project directory, named exactly like this:

```text
2015_rsh_payments.csv
2016_rsh_payments.csv
2017_rsh_payments.csv
```

The year in the filename must match the year you pass on the command line.

## Usage

Run from the repository root.

### Single Physician

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
python3 research_extractor.py \
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
python3 -m pip install pandas tabulate
```
