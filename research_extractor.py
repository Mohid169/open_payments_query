import pandas as pd
import os
import logging
import re
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_research_payments_for_physician(
    csv_path,
    physician_last_name,
    physician_first_name,
    physician_middle=None,
    case_sensitive=False,
):
    """
    Reads the CMS Open Payments Research CSV file and returns payment information for a physician.

    The function:
    1. Filters rows where the provided name appears as a Covered Recipient or Principal Investigator
    2. Handles optional middle name/initial matching
    3. Provides case-sensitive or case-insensitive name matching
    4. Determines NPI (National Provider Identifier) from available columns
    5. Groups and sums payment amounts by NPI or returns an overall sum

    Parameters:
    csv_path (str): Path to the CMS Open Payments Research CSV file
    physician_last_name (str): Physician's last name
    physician_first_name (str): Physician's first name
    physician_middle (str, optional): Physician's middle name or initial
    case_sensitive (bool, optional): Whether to perform case-sensitive name matching

    Returns:
    tuple or float: (DataFrame with NPIs and associated total payments, dict of entry counts per NPI),
                    or a float if no NPIs found

    Raises:
    FileNotFoundError: If the CSV file doesn't exist
    ValueError: If required columns are missing from the CSV
    """
    # Validate file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Reading CMS Open Payments file: {csv_path}")
    try:
        # Sample a few rows to check columns and determine optimal chunk size
        sample_df = pd.read_csv(csv_path, nrows=5, dtype=str)

        # Check which column naming convention is being used
        if (
            "Physician_First_Name" in sample_df.columns
            and "Physician_Last_Name" in sample_df.columns
        ):
            # Using pre-2016 naming convention
            logger.info("Detected pre-2016 column format (Physician_* prefix)")
            first_name_col = "Physician_First_Name"
            middle_name_col = "Physician_Middle_Name"
            last_name_col = "Physician_Last_Name"
            npi_col = "Physician_NPI"
            profile_id_col = "Physician_Profile_ID"
            # Specialty columns might be different
            specialty_prefix = "Physician_Specialty"
            pi_prefix = "Principal_Investigator_"
        else:
            # Using 2016+ naming convention
            logger.info("Detected 2016+ column format (Covered_Recipient_* prefix)")
            first_name_col = "Covered_Recipient_First_Name"
            middle_name_col = "Covered_Recipient_Middle_Name"
            last_name_col = "Covered_Recipient_Last_Name"
            npi_col = "Covered_Recipient_NPI"
            profile_id_col = "Covered_Recipient_Profile_ID"
            specialty_prefix = "Covered_Recipient_Specialty"
            pi_prefix = "Principal_Investigator_"

        # Always check for payment amount column
        payment_col = "Total_Amount_of_Payment_USDollars"

        # Validate required columns exist
        required_cols = [payment_col, first_name_col, last_name_col]
        missing_cols = [col for col in required_cols if col not in sample_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in CSV: {', '.join(missing_cols)}"
            )

        # Read CSV file with appropriate settings for large files
        # Using dtype=str for all columns initially to prevent type inference issues
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)

        # Convert payment column to numeric with proper error handling
        df[payment_col] = pd.to_numeric(df[payment_col], errors="coerce")

        # Filter out rows with NaN payment amounts
        initial_count = len(df)
        df = df.dropna(subset=[payment_col])
        if len(df) < initial_count:
            logger.warning(
                f"Dropped {initial_count - len(df)} rows with invalid payment amounts"
            )

        # Standardize input names based on case sensitivity setting
        if case_sensitive:
            last_name_std = physician_last_name.strip()
            first_name_std = physician_first_name.strip()
            middle_std = physician_middle.strip() if physician_middle else None

            # Define comparison functions for case-sensitive matching
            def match_last(x):
                return x == last_name_std if pd.notna(x) else False

            def match_first(x):
                return x == first_name_std if pd.notna(x) else False

            def match_middle(x, middle):
                # Handle middle initial (single character with possible period)
                if middle and len(middle) <= 2 and "." in middle:
                    return (
                        x.startswith(middle.replace(".", "")) if pd.notna(x) else False
                    )
                return x == middle if pd.notna(x) else False

        else:
            # Case-insensitive matching
            last_name_std = physician_last_name.strip().lower()
            first_name_std = physician_first_name.strip().lower()
            middle_std = physician_middle.strip().lower() if physician_middle else None

            # Define comparison functions for case-insensitive matching
            def match_last(x):
                return x.lower() == last_name_std if pd.notna(x) else False

            def match_first(x):
                return x.lower() == first_name_std if pd.notna(x) else False

            def match_middle(x, middle):
                # Handle middle initial (single character with possible period)
                if middle and len(middle) <= 2 and "." in middle:
                    return (
                        x.lower().startswith(middle.replace(".", "").lower())
                        if pd.notna(x)
                        else False
                    )
                return x.lower() == middle if pd.notna(x) else False

        # Mask for when physician appears as Covered Recipient/Physician
        df["match_covered_last"] = df[last_name_col].apply(match_last)
        df["match_covered_first"] = df[first_name_col].apply(match_first)

        # Handle middle name/initial if provided and column exists
        if middle_std and middle_name_col in df.columns:
            df["match_covered_middle"] = df[middle_name_col].apply(
                lambda x: match_middle(x, middle_std)
            )
            mask_covered = (
                df["match_covered_last"]
                & df["match_covered_first"]
                & df["match_covered_middle"]
            )
        else:
            mask_covered = df["match_covered_last"] & df["match_covered_first"]

        # Mask for when physician appears as Principal Investigator (PI1-PI5)
        mask_pi = pd.Series([False] * len(df))

        for i in range(1, 6):
            pi_last_col = f"{pi_prefix}{i}_Last_Name"
            pi_first_col = f"{pi_prefix}{i}_First_Name"
            pi_middle_col = f"{pi_prefix}{i}_Middle_Name"

            if pi_last_col in df.columns and pi_first_col in df.columns:
                df[f"match_pi{i}_last"] = df[pi_last_col].apply(match_last)
                df[f"match_pi{i}_first"] = df[pi_first_col].apply(match_first)

                # Handle middle name/initial for PI if provided and column exists
                if middle_std and pi_middle_col in df.columns:
                    df[f"match_pi{i}_middle"] = df[pi_middle_col].apply(
                        lambda x: match_middle(x, middle_std)
                    )
                    current_mask = (
                        df[f"match_pi{i}_last"]
                        & df[f"match_pi{i}_first"]
                        & df[f"match_pi{i}_middle"]
                    )
                else:
                    current_mask = df[f"match_pi{i}_last"] & df[f"match_pi{i}_first"]

                mask_pi = mask_pi | current_mask

        # Combine both masks
        combined_mask = mask_covered | mask_pi
        filtered_df = df[combined_mask].copy()

        # Log statistics
        logger.info(
            f"Found {len(filtered_df)} entries matching '{first_name_std} {middle_std or ''} {last_name_std}'"
        )

        # Clean up temporary matching columns
        match_cols = [col for col in filtered_df.columns if col.startswith("match_")]
        filtered_df = filtered_df.drop(columns=match_cols)

        # Handle case where no matching records found
        if len(filtered_df) == 0:
            logger.warning(
                f"No payments found for {first_name_std} {middle_std or ''} {last_name_std}"
            )
            return 0.0

        # Create a new column 'NPI' to help disambiguate duplicates
        def get_npi(row):
            # First try Covered Recipient/Physician NPI
            if pd.notna(row.get(npi_col)) and row.get(npi_col) != "":
                return row.get(npi_col)

            # Then try each Principal Investigator NPI
            for i in range(1, 6):
                pi_npi_col = f"{pi_prefix}{i}_NPI"
                if (
                    pi_npi_col in row
                    and pd.notna(row[pi_npi_col])
                    and row[pi_npi_col] != ""
                ):
                    return row[pi_npi_col]

            # If no NPI found, try to create a composite identifier from name and profile ID
            if pd.notna(row.get(profile_id_col)) and row.get(profile_id_col) != "":
                return f"PROFILE_{row.get(profile_id_col)}"

            # Last resort: return Unknown with a row index to keep them separate
            return f"UNKNOWN_{row.name}"

        # Apply the function to create an NPI column
        filtered_df["NPI"] = filtered_df.apply(get_npi, axis=1)

        # Group by NPI and sum payments
        group_totals = filtered_df.groupby("NPI")[payment_col].sum().reset_index()

        # Add useful information to the result
        if len(group_totals) > 0:
            # Enrich with physician names and specialties where available
            npi_info = {}
            for npi in group_totals["NPI"]:
                # Get first row for this NPI
                row = filtered_df[filtered_df["NPI"] == npi].iloc[0]

                # Build name from covered recipient or principal investigator fields
                if pd.notna(row.get(first_name_col)):
                    name = f"{row.get(first_name_col)} "
                    if pd.notna(row.get(middle_name_col)):
                        name += f"{row.get(middle_name_col)} "
                    name += row.get(last_name_col)

                    # Get all specialties - handle both old and new format
                    specialties = []

                    # Check for numbered specialty columns (typical in newer format)
                    specialties_found = False
                    for i in range(1, 6):
                        specialty_col = f"{specialty_prefix}_{i}"
                        if (
                            specialty_col in row
                            and pd.notna(row.get(specialty_col))
                            and row.get(specialty_col).strip() != ""
                        ):
                            specialties.append(row.get(specialty_col).strip())
                            specialties_found = True

                    # If no numbered specialties found, check for single specialty column (typical in older format)
                    if not specialties_found and specialty_prefix in row:
                        if (
                            pd.notna(row.get(specialty_prefix))
                            and row.get(specialty_prefix).strip() != ""
                        ):
                            specialties.append(row.get(specialty_prefix).strip())

                    # Join specialties or use 'Unknown' if none found
                    specialty = "; ".join(specialties) if specialties else "Unknown"
                else:
                    # Try to get name from PI fields
                    for i in range(1, 6):
                        first_col = f"{pi_prefix}{i}_First_Name"
                        middle_col = f"{pi_prefix}{i}_Middle_Name"
                        last_col = f"{pi_prefix}{i}_Last_Name"

                        if (
                            first_col in row
                            and last_col in row
                            and pd.notna(row.get(first_col))
                            and pd.notna(row.get(last_col))
                        ):
                            name = f"{row.get(first_col)} "
                            if middle_col in row and pd.notna(row.get(middle_col)):
                                name += f"{row.get(middle_col)} "
                            name += row.get(last_col)
                            specialty = "Principal Investigator"
                            break
                    else:
                        name = "Unknown"
                        specialty = "Unknown"

                npi_info[npi] = {"Physician_Name": name, "Specialty": specialty}

            # Add physician name and specialty columns
            group_totals["Physician_Name"] = group_totals["NPI"].map(
                lambda x: npi_info[x]["Physician_Name"]
            )
            group_totals["Specialty"] = group_totals["NPI"].map(
                lambda x: npi_info[x]["Specialty"]
            )

            # Rename payment column for clarity
            group_totals.rename(
                columns={payment_col: "Total_Payment_USD"}, inplace=True
            )

            # Reorder columns for better readability
            group_totals = group_totals[
                ["NPI", "Physician_Name", "Specialty", "Total_Payment_USD"]
            ]

            # Count original entries per NPI
            npi_entry_counts = {}
            for npi in group_totals["NPI"]:
                npi_entry_counts[npi] = len(filtered_df[filtered_df["NPI"] == npi])

            logger.info(f"Grouped into {len(group_totals)} unique physicians")
            return group_totals, npi_entry_counts
        else:
            # If grouping failed for some reason, return the overall sum
            total_payment = filtered_df[payment_col].sum()
            logger.info(f"Returning total payment amount: ${total_payment:,.2f}")
            return total_payment

    except Exception as e:
        logger.error(f"Error processing CMS Open Payments file: {str(e)}")
        raise


def parse_name(full_name):
    """
    Parse a full name into first, middle, and last name components.
    Handles various formats and special cases.

    Parameters:
    full_name (str): Full name to parse

    Returns:
    tuple: (first_name, middle_name, last_name)
    """
    # Remove any content in parentheses (like city/state information)
    full_name = re.sub(r"\([^)]*\)", "", full_name).strip()

    # Split the name into parts
    parts = full_name.split()

    # Handle simple cases
    if len(parts) == 2:
        # Just first and last name
        return parts[0], None, parts[1]

    elif len(parts) == 3:
        # Standard first, middle, last format
        return parts[0], parts[1], parts[2]

    elif len(parts) > 3:
        # More complex cases - assume first name is first part,
        # last name is last part, and everything in between is middle name(s)
        first_name = parts[0]
        last_name = parts[-1]
        middle_name = " ".join(parts[1:-1])
        return first_name, middle_name, last_name

    else:
        # Just return what we have, handle edge cases
        if len(parts) == 1:
            return None, None, parts[0]  # Just last name
        else:
            return None, None, full_name  # Unknown format


def read_names_from_file(filename):
    """
    Read a list of names from a text file.

    Parameters:
    filename (str): Path to the text file containing names

    Returns:
    list: List of names from the file
    """
    names = []
    try:
        with open(filename, "r") as file:
            for line in file:
                name = line.strip()
                if name:  # Skip empty lines
                    names.append(name)
        logger.info(f"Successfully read {len(names)} names from {filename}")
        return names
    except Exception as e:
        logger.error(f"Error reading names file: {str(e)}")
        return []


def process_physician(
    first_name, middle_name, last_name, years_to_process, case_sensitive=False
):
    """
    Process research payments for a single physician across multiple years.

    Parameters:
    first_name (str): Physician's first name
    middle_name (str): Physician's middle name or initial (can be None)
    last_name (str): Physician's last name
    years_to_process (list): List of years to process
    case_sensitive (bool): Whether to use case-sensitive matching

    Returns:
    tuple: (combined_df, total_payment, total_entries)
    """
    # Dictionary to store results by year
    results_by_year = {}
    all_npis = set()

    # Dictionary to store entry counts by NPI
    all_entry_counts = {}

    # Process each year
    for year in years_to_process:
        csv_file_path = f"{year}_rsh_payments.csv"
        logger.info(f"Processing data for year {year} from file {csv_file_path}")

        try:
            # If middle name is provided, check both with and without
            if middle_name:
                logger.info(f"Checking with middle name: {middle_name}")
                result_with_middle = get_research_payments_for_physician(
                    csv_file_path,
                    last_name,
                    first_name,
                    physician_middle=middle_name,
                    case_sensitive=case_sensitive,
                )

                logger.info("Checking without middle name")
                result_without_middle = get_research_payments_for_physician(
                    csv_file_path,
                    last_name,
                    first_name,
                    physician_middle=None,
                    case_sensitive=case_sensitive,
                )

                # Combine results from both searches
                if isinstance(result_with_middle, tuple) and isinstance(
                    result_without_middle, tuple
                ):
                    # Both returned DataFrames
                    df_with_middle, counts_with_middle = result_with_middle
                    df_without_middle, counts_without_middle = result_without_middle

                    # Combine the DataFrames, removing duplicates based on NPI
                    combined_df = pd.concat(
                        [df_with_middle, df_without_middle]
                    ).drop_duplicates(subset=["NPI"])

                    # Merge entry counts (taking the larger count if duplicates exist)
                    combined_counts = counts_with_middle.copy()
                    for npi, count in counts_without_middle.items():
                        if npi in combined_counts:
                            combined_counts[npi] = max(combined_counts[npi], count)
                        else:
                            combined_counts[npi] = count

                    result = (combined_df, combined_counts)

                elif isinstance(result_with_middle, tuple):
                    # Only with_middle search returned DataFrame
                    result = result_with_middle

                elif isinstance(result_without_middle, tuple):
                    # Only without_middle search returned DataFrame
                    result = result_without_middle

                else:
                    # Both returned scalar values, use the sum
                    result = result_with_middle + result_without_middle
            else:
                # No middle name provided, just check without
                result = get_research_payments_for_physician(
                    csv_file_path,
                    last_name,
                    first_name,
                    physician_middle=None,
                    case_sensitive=case_sensitive,
                )

            # Handle the result
            if isinstance(result, tuple):
                # Unpack the DataFrame and entry counts
                df_result, entry_counts = result
                results_by_year[year] = df_result

                # Update the global entry counts
                for npi, count in entry_counts.items():
                    all_entry_counts[npi] = all_entry_counts.get(npi, 0) + count

                # Collect all unique NPIs
                all_npis.update(df_result["NPI"].tolist())
            else:
                # Handle scalar result (no matching records found)
                results_by_year[year] = result

        except Exception as e:
            logger.error(f"Error processing {year} data: {str(e)}")
            # Continue with other years even if one fails
            results_by_year[year] = None

    # Create combined multi-year report if we have DataFrame results
    if any(isinstance(result, pd.DataFrame) for result in results_by_year.values()):
        # Initialize a DataFrame to hold combined data
        columns = ["NPI", "Physician_Name", "Specialty", "Entry_Count"]
        for year in years_to_process:
            columns.append(f"Payment_{year}_USD")
        columns.append("Total_USD")

        combined_data = []

        # For each unique NPI across all years
        for npi in all_npis:
            row_data = {
                "NPI": npi,
                "Total_USD": 0.0,
                "Entry_Count": all_entry_counts.get(
                    npi, 0
                ),  # Use the collected entry counts
            }

            # Get physician name and specialty from the first year that has this NPI
            for year in years_to_process:
                result = results_by_year[year]
                if isinstance(result, pd.DataFrame) and npi in result["NPI"].values:
                    npi_data = result[result["NPI"] == npi].iloc[0]
                    row_data["Physician_Name"] = npi_data["Physician_Name"]
                    row_data["Specialty"] = npi_data["Specialty"]
                    break
            else:
                # If we didn't find a name or specialty (shouldn't happen), use placeholders
                row_data["Physician_Name"] = "Unknown"
                row_data["Specialty"] = "Unknown"

            # Fill in payment data for each year
            for year in years_to_process:
                result = results_by_year[year]
                payment = 0.0

                if isinstance(result, pd.DataFrame) and npi in result["NPI"].values:
                    payment = result[result["NPI"] == npi].iloc[0]["Total_Payment_USD"]

                row_data[f"Payment_{year}_USD"] = payment
                row_data["Total_USD"] += payment

            combined_data.append(row_data)

        # Create DataFrame and sort by total payment descending
        combined_df = pd.DataFrame(combined_data)
        combined_df = combined_df.sort_values("Total_USD", ascending=False)

        # Calculate totals
        grand_total = combined_df["Total_USD"].sum()
        total_entries = combined_df["Entry_Count"].sum()

        return combined_df, grand_total, total_entries
    else:
        # If no DataFrame results, return a simple total
        total_across_years = sum(
            result
            for result in results_by_year.values()
            if isinstance(result, (int, float))
        )
        return None, total_across_years, 0


def process_physician_list(
    names_file, years_to_process, output_dir=None, case_sensitive=False
):
    """
    Process a list of physicians from a file and save results to CSV files.

    Parameters:
    names_file (str): Path to the file containing physician names
    years_to_process (list): List of years to process
    output_dir (str): Directory to save output files (if None, use current directory)
    case_sensitive (bool): Whether to use case-sensitive name matching
    """
    # Read names from file
    names = read_names_from_file(names_file)
    if not names:
        logger.error("No names found in the input file.")
        return

    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare summary data
    years_string = (
        f"{min(years_to_process)}-{max(years_to_process)}"
        if len(years_to_process) > 1
        else str(years_to_process[0])
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"payment_summary_{years_string}_{timestamp}.csv"
    if output_dir:
        summary_filename = os.path.join(output_dir, summary_filename)

    summary_data = []

    # Process each physician
    for i, name in enumerate(names):
        logger.info(f"Processing physician {i+1}/{len(names)}: {name}")

        # Parse the name
        first_name, middle_name, last_name = parse_name(name)
        if not first_name or not last_name:
            logger.warning(f"Could not parse name properly: {name}")
            continue

        # Log the parsed name components
        logger.info(
            f"Parsed name: First: {first_name}, Middle: {middle_name}, Last: {last_name}"
        )

        try:
            # Process this physician
            combined_df, total_payment, total_entries = process_physician(
                first_name, middle_name, last_name, years_to_process, case_sensitive
            )

            # Initialize summary entry with basic information
            summary_entry = {
                "Full_Name": name,
                "First_Name": first_name,
                "Middle_Name": middle_name or "",
                "Last_Name": last_name,
                "Total_Payment": total_payment,
                "Total_Entries": total_entries,
            }

            # Add year-by-year breakdowns
            # If we have detailed results, get the yearly totals
            if combined_df is not None and not combined_df.empty:
                for year in years_to_process:
                    year_col = f"Payment_{year}_USD"
                    if year_col in combined_df.columns:
                        summary_entry[year_col] = combined_df[year_col].sum()
                    else:
                        summary_entry[year_col] = 0.0
            else:
                # If we don't have detailed results, we can't break down by year
                for year in years_to_process:
                    summary_entry[f"Payment_{year}_USD"] = 0.0

            summary_data.append(summary_entry)

            # If we got detailed results, save them to a separate CSV
            if combined_df is not None and not combined_df.empty:
                # Create a filename based on the physician's name
                safe_name = re.sub(r"[^\w\s]", "", name).replace(" ", "_")
                detail_filename = f"research_payments_{safe_name}_{years_string}.csv"
                if output_dir:
                    detail_filename = os.path.join(output_dir, detail_filename)

                # Format currency values without the dollar sign for CSV
                export_df = combined_df.copy()
                for year in years_to_process:
                    export_df[f"Payment_{year}_USD"] = export_df[
                        f"Payment_{year}_USD"
                    ].apply(lambda x: f"{x:.2f}")
                export_df["Total_USD"] = export_df["Total_USD"].apply(
                    lambda x: f"{x:.2f}"
                )

                # Save to CSV
                export_df.to_csv(detail_filename, index=False)
                logger.info(f"Saved detailed results to: {detail_filename}")

                # Print a summary for this physician
                print(f"\nResults for {name}:")
                print(f"Total Payment: ${total_payment:,.2f}")
                print(f"Total Entries: {total_entries}")
                print(f"Saved to: {detail_filename}")
            else:
                print(
                    f"\nNo detailed results for {name}. Total payment: ${total_payment:,.2f}"
                )

        except Exception as e:
            logger.error(f"Error processing physician {name}: {str(e)}")
            # Add error entry with zero values for all years
            error_entry = {
                "Full_Name": name,
                "First_Name": first_name,
                "Middle_Name": middle_name or "",
                "Last_Name": last_name,
                "Total_Payment": 0,
                "Total_Entries": 0,
                "Error": str(e),
            }
            for year in years_to_process:
                error_entry[f"Payment_{year}_USD"] = 0.0

            summary_data.append(error_entry)

    # Save summary data to CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Reorder columns to put years in sequence
        base_cols = ["Full_Name", "First_Name", "Middle_Name", "Last_Name"]
        year_cols = [f"Payment_{year}_USD" for year in sorted(years_to_process)]
        end_cols = ["Total_Payment", "Total_Entries"]
        error_col = ["Error"] if "Error" in summary_df.columns else []

        col_order = base_cols + year_cols + end_cols + error_col
        summary_df = summary_df[col_order]

        # Sort by total payment descending
        summary_df = summary_df.sort_values("Total_Payment", ascending=False)

        # Format values for better readability
        for year in years_to_process:
            col = f"Payment_{year}_USD"
            summary_df[col] = summary_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "0.00"
            )

        summary_df["Total_Payment"] = summary_df["Total_Payment"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "0.00"
        )

        # Save to CSV
        summary_df.to_csv(summary_filename, index=False)
        logger.info(f"Saved summary results to: {summary_filename}")

        # Print the overall summary
        print("\nOverall Summary:")
        print(f"Total Physicians Processed: {len(summary_data)}")

        # Calculate totals for each year
        year_totals = {}
        for year in years_to_process:
            col = f"Payment_{year}_USD"
            # Convert back to float for summing
            year_totals[year] = sum(float(x) for x in summary_df[col] if pd.notnull(x))
            print(f"Total Payments for {year}: ${year_totals[year]:,.2f}")

        print(
            f"Total Payments Across All Years: ${summary_df['Total_Payment'].astype(float).sum():,.2f}"
        )
        print(f"Total Entries: {summary_df['Total_Entries'].sum()}")
        print(f"Summary saved to: {summary_filename}")
    else:
        logger.warning("No summary data to save.")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract research payments for physicians from CMS Open Payments data."
    )

    # Add arguments
    parser.add_argument(
        "--names_file", type=str, help="Path to file containing physician names"
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2015, 2016], help="Years to process"
    )
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument(
        "--case_sensitive", action="store_true", help="Use case-sensitive name matching"
    )

    # Add single physician mode arguments
    parser.add_argument(
        "--first_name", type=str, help="Physician first name (for single mode)"
    )
    parser.add_argument(
        "--middle_name", type=str, help="Physician middle name (for single mode)"
    )
    parser.add_argument(
        "--last_name", type=str, help="Physician last name (for single mode)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if we should run in batch mode or single physician mode
    if args.names_file:
        # Batch mode - process list of physicians from file
        logger.info(f"Running in batch mode with names from: {args.names_file}")
        process_physician_list(
            args.names_file, args.years, args.output_dir, args.case_sensitive
        )
    elif args.first_name and args.last_name:
        # Single physician mode
        logger.info(
            f"Running in single physician mode for: {args.first_name} {args.middle_name or ''} {args.last_name}"
        )

        # Process the physician
        combined_df, total_payment, total_entries = process_physician(
            args.first_name,
            args.middle_name,
            args.last_name,
            args.years,
            args.case_sensitive,
        )

        # Print results
        print(
            f"\nResearch payments for {args.first_name} {args.middle_name or ''} {args.last_name}:"
        )
        print("-" * 80)
        print(f"Total Payment: ${total_payment:,.2f}")
        print(f"Total Entries: {total_entries}")

        # Save results if we have detailed data
        if combined_df is not None and not combined_df.empty:
            # Set up display options
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)
            pd.set_option("display.float_format", "${:,.2f}".format)

            # Print detailed results
            print("\nDetailed Results:")
            print("-" * 80)
            print(combined_df.to_string(index=False))

            # Create filename and save to CSV
            years_string = (
                f"{min(args.years)}-{max(args.years)}"
                if len(args.years) > 1
                else str(args.years[0])
            )
            output_filename = f"research_payments_{args.last_name}_{args.first_name}_{years_string}.csv"
            if args.output_dir:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                output_filename = os.path.join(args.output_dir, output_filename)

            # Format for CSV
            export_df = combined_df.copy()
            for year in args.years:
                export_df[f"Payment_{year}_USD"] = export_df[
                    f"Payment_{year}_USD"
                ].apply(lambda x: f"{x:.2f}")
            export_df["Total_USD"] = export_df["Total_USD"].apply(lambda x: f"{x:.2f}")

            # Save to CSV
            export_df.to_csv(output_filename, index=False)
            print(f"\nResults saved to: {output_filename}")
    else:
        # Default example - using hardcoded values
        print("No arguments provided. Running with example values:")
        first_name_to_find = "Benjamin"
        middle_to_find = "G"
        last_name_to_find = "Domb"
        years_to_process = [2015, 2016]

        print(
            f"Physician: {first_name_to_find} {middle_to_find or ''} {last_name_to_find}"
        )
        print(f"Years: {years_to_process}")

        combined_df, total_payment, total_entries = process_physician(
            first_name_to_find,
            middle_to_find,
            last_name_to_find,
            years_to_process,
            False,
        )

        if combined_df is not None:
            # Set up display options
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)
            pd.set_option("display.float_format", "${:,.2f}".format)

            # Print detailed results
            print("\nDetailed Results:")
            print("-" * 80)
            print(combined_df.to_string(index=False))

            # Create filename and save to CSV
            years_string = (
                f"{min(years_to_process)}-{max(years_to_process)}"
                if len(years_to_process) > 1
                else str(years_to_process[0])
            )
            output_filename = f"research_payments_{last_name_to_find}_{first_name_to_find}_{years_string}.csv"

            # Format for CSV
            export_df = combined_df.copy()
            for year in years_to_process:
                export_df[f"Payment_{year}_USD"] = export_df[
                    f"Payment_{year}_USD"
                ].apply(lambda x: f"{x:.2f}")
            export_df["Total_USD"] = export_df["Total_USD"].apply(lambda x: f"{x:.2f}")

            # Save to CSV
            export_df.to_csv(output_filename, index=False)
            print(f"\nResults saved to: {output_filename}")
        else:
            print(f"\nNo detailed results. Total payment: ${total_payment:,.2f}")
