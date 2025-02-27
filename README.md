# open_payments_query
# Research Payments Extractor

A Python script to analyze and summarize research payments data from the CMS Open Payments database across multiple years.

## Features

- Searches for physicians by name across multiple years of payment data
- Supports fuzzy name matching to handle variations in physician names
- Validates physician identity using NPI (National Provider Identifier)
- Aggregates payment data across years
- Provides detailed payment summaries including:
  - Physician name and specialty
  - Payment amounts by year
  - Total payments across all years
- Sorts results by total payment amount
- Optional CSV export functionality

## Usage
1. Download CMS Open Payments Research Payment data files for desired years from the [CMS website](https://www.cms.gov/OpenPayments/Data/Dataset-Downloads)

2. Place the downloaded CSV files in a directory

3. Run the script with required arguments:

```
python research_extractor.py --names_file physicians.txt --years 2015 2016 --output_dir results
```
or  
```
python research_extractor.py --first_name Benjamin --middle_name G --last_name Domb --years 2015 2016
```

4. The script will output the total payment amount and a list of NPIs associated with the physician.

5. Optionally, the script can export the payment data to a CSV file.
