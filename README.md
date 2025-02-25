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

