import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_research_payments_for_physician(
    csv_path, 
    physician_last_name, 
    physician_first_name, 
    physician_middle=None,
    case_sensitive=False
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
    pd.DataFrame or float: DataFrame with NPIs and associated total payments, or a float if no NPIs found
    
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
        required_cols = ['Total_Amount_of_Payment_USDollars', 
                        'Covered_Recipient_First_Name', 
                        'Covered_Recipient_Last_Name']
        
        # Validate required columns exist
        missing_cols = [col for col in required_cols if col not in sample_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")
        
        # Read CSV file with appropriate settings for large files
        # Using dtype=str for all columns initially to prevent type inference issues
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        
        # Convert payment column to numeric with proper error handling
        df['Total_Amount_of_Payment_USDollars'] = pd.to_numeric(
            df['Total_Amount_of_Payment_USDollars'], 
            errors='coerce'
        )
        
        # Filter out rows with NaN payment amounts
        initial_count = len(df)
        df = df.dropna(subset=['Total_Amount_of_Payment_USDollars'])
        if len(df) < initial_count:
            logger.warning(f"Dropped {initial_count - len(df)} rows with invalid payment amounts")
            
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
                if middle and len(middle) <= 2 and '.' in middle:
                    return x.startswith(middle.replace('.', '')) if pd.notna(x) else False
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
                if middle and len(middle) <= 2 and '.' in middle:
                    return x.lower().startswith(middle.replace('.', '').lower()) if pd.notna(x) else False
                return x.lower() == middle if pd.notna(x) else False
        
        # Mask for when physician appears as Covered Recipient
        df['match_covered_last'] = df['Covered_Recipient_Last_Name'].apply(match_last)
        df['match_covered_first'] = df['Covered_Recipient_First_Name'].apply(match_first)
        
        # Handle middle name/initial for Covered Recipient if provided and column exists
        if middle_std and 'Covered_Recipient_Middle_Name' in df.columns:
            df['match_covered_middle'] = df['Covered_Recipient_Middle_Name'].apply(
                lambda x: match_middle(x, middle_std)
            )
            mask_covered = df['match_covered_last'] & df['match_covered_first'] & df['match_covered_middle']
        else:
            mask_covered = df['match_covered_last'] & df['match_covered_first']
        
        # Mask for when physician appears as Principal Investigator (PI1-PI5)
        mask_pi = pd.Series([False] * len(df))
        
        for i in range(1, 6):
            pi_last_col = f'Principal_Investigator_{i}_Last_Name'
            pi_first_col = f'Principal_Investigator_{i}_First_Name'
            pi_middle_col = f'Principal_Investigator_{i}_Middle_Name'
            
            if pi_last_col in df.columns and pi_first_col in df.columns:
                df[f'match_pi{i}_last'] = df[pi_last_col].apply(match_last)
                df[f'match_pi{i}_first'] = df[pi_first_col].apply(match_first)
                
                # Handle middle name/initial for PI if provided and column exists
                if middle_std and pi_middle_col in df.columns:
                    df[f'match_pi{i}_middle'] = df[pi_middle_col].apply(
                        lambda x: match_middle(x, middle_std)
                    )
                    current_mask = (
                        df[f'match_pi{i}_last'] & 
                        df[f'match_pi{i}_first'] & 
                        df[f'match_pi{i}_middle']
                    )
                else:
                    current_mask = df[f'match_pi{i}_last'] & df[f'match_pi{i}_first']
                
                mask_pi = mask_pi | current_mask
        
        # Combine both masks
        combined_mask = mask_covered | mask_pi
        filtered_df = df[combined_mask].copy()
        
        # Log statistics
        logger.info(f"Found {len(filtered_df)} entries matching '{first_name_std} {middle_std or ''} {last_name_std}'")
        
        # Clean up temporary matching columns
        match_cols = [col for col in filtered_df.columns if col.startswith('match_')]
        filtered_df = filtered_df.drop(columns=match_cols)
        
        # Handle case where no matching records found
        if len(filtered_df) == 0:
            logger.warning(f"No payments found for {first_name_std} {middle_std or ''} {last_name_std}")
            return 0.0
        
        # Create a new column 'NPI' to help disambiguate duplicates
        def get_npi(row):
            # First try Covered Recipient NPI
            if pd.notna(row.get('Covered_Recipient_NPI')) and row.get('Covered_Recipient_NPI') != '':
                return row.get('Covered_Recipient_NPI')
            
            # Then try each Principal Investigator NPI
            for i in range(1, 6):
                pi_npi_col = f'Principal_Investigator_{i}_NPI'
                if pi_npi_col in row and pd.notna(row[pi_npi_col]) and row[pi_npi_col] != '':
                    return row[pi_npi_col]
            
            # If no NPI found, try to create a composite identifier from name and profile ID
            if pd.notna(row.get('Covered_Recipient_Profile_ID')) and row.get('Covered_Recipient_Profile_ID') != '':
                return f"PROFILE_{row.get('Covered_Recipient_Profile_ID')}"
                
            # Last resort: return Unknown with a row index to keep them separate
            return f"UNKNOWN_{row.name}"
        
        # Apply the function to create an NPI column
        filtered_df['NPI'] = filtered_df.apply(get_npi, axis=1)
        
        # Group by NPI and sum payments
        group_totals = filtered_df.groupby('NPI')['Total_Amount_of_Payment_USDollars'].sum().reset_index()
        
                    # Add useful information to the result
        if len(group_totals) > 0:
            # Enrich with physician names and specialties where available
            npi_info = {}
            for npi in group_totals['NPI']:
                # Get first row for this NPI
                row = filtered_df[filtered_df['NPI'] == npi].iloc[0]
                
                # Build name from covered recipient or principal investigator fields
                if pd.notna(row.get('Covered_Recipient_First_Name')):
                    name = f"{row.get('Covered_Recipient_First_Name')} "
                    if pd.notna(row.get('Covered_Recipient_Middle_Name')):
                        name += f"{row.get('Covered_Recipient_Middle_Name')} "
                    name += row.get('Covered_Recipient_Last_Name')
                    
                    # Get all specialties (typically up to 5 in CMS data)
                    specialties = []
                    for i in range(1, 6):
                        specialty_col = f'Covered_Recipient_Specialty_{i}'
                        if specialty_col in row and pd.notna(row.get(specialty_col)) and row.get(specialty_col).strip() != '':
                            specialties.append(row.get(specialty_col).strip())
                    
                    # Join specialties or use 'Unknown' if none found
                    specialty = '; '.join(specialties) if specialties else 'Unknown'
                else:
                    # Try to get name from PI fields
                    for i in range(1, 6):
                        first_col = f'Principal_Investigator_{i}_First_Name'
                        middle_col = f'Principal_Investigator_{i}_Middle_Name'
                        last_col = f'Principal_Investigator_{i}_Last_Name'
                        
                        if pd.notna(row.get(first_col)) and pd.notna(row.get(last_col)):
                            name = f"{row.get(first_col)} "
                            if pd.notna(row.get(middle_col)):
                                name += f"{row.get(middle_col)} "
                            name += row.get(last_col)
                            specialty = 'Principal Investigator'
                            break
                    else:
                        name = 'Unknown'
                        specialty = 'Unknown'
                
                npi_info[npi] = {'Physician_Name': name, 'Specialty': specialty}
            
            # Add physician name and specialty columns
            group_totals['Physician_Name'] = group_totals['NPI'].map(lambda x: npi_info[x]['Physician_Name'])
            group_totals['Specialty'] = group_totals['NPI'].map(lambda x: npi_info[x]['Specialty'])
            
            # Rename payment column for clarity
            group_totals.rename(columns={'Total_Amount_of_Payment_USDollars': 'Total_Payment_USD'}, inplace=True)
            
            # Reorder columns for better readability
            group_totals = group_totals[['NPI', 'Physician_Name', 'Specialty', 'Total_Payment_USD']]
            
            logger.info(f"Grouped into {len(group_totals)} unique physicians")
            return group_totals
        else:
            # If grouping failed for some reason, return the overall sum
            total_payment = filtered_df['Total_Amount_of_Payment_USDollars'].sum()
            logger.info(f"Returning total payment amount: ${total_payment:,.2f}")
            return total_payment
            
    except Exception as e:
        logger.error(f"Error processing CMS Open Payments file: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage with configuration for robust file processing
    csv_file_path = '2018_rsh_payments.csv'
    
    # Query parameters
    first_name_to_find = "Jonathan"
    middle_to_find = "Matthew"  # Optional, set to None if not needed
    last_name_to_find = "Vigdorchik"
    
    try:
        result = get_research_payments_for_physician(
            csv_file_path, 
            last_name_to_find,
            first_name_to_find, 
            physician_middle=middle_to_find,
            case_sensitive=False  # Set to True for exact case matching
        )
        
        if isinstance(result, pd.DataFrame):
            print("\nPayment totals grouped by physician:")
            print(result.to_string(index=False))
            print(f"\nTotal research payments: ${result['Total_Payment_USD'].sum():,.2f}")
        else:
            print(f"\nTotal research payments for {first_name_to_find} {middle_to_find or ''} {last_name_to_find}: ${result:,.2f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")