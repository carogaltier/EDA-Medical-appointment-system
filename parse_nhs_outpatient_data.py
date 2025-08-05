## =====================================================================================
# NHS Outpatient Data Parser (Optional)
# =====================================================================================
# Description:
# This utility function loads and processes outpatient statistics from NHS England (2023–24).
# It was used to derive the default parameters for:
# - DEFAULT_AGE_GENDER_PROBS         ← Summary Report 3
# - DEFAULT_FIRST_ATTENDANCE_RATIO   ← Summary Report 2
# - DEFAULT_STATUS_RATES             ← Summary Report 1
#
# The function is included for transparency and reproducibility but is NOT required at runtime.
# It can be used manually to refresh the constant values in this module.
#
# Dependencies: pandas
#
# -------------------------------------------------------------------------------------
# Author: Carolina Gonzalez Galtier
# Created: August 2025
# License: CC BY 4.0
# Repository: https://github.com/carogaltier/Synthetic-Medical-Appointment-Dataset
# =====================================================================================


import pandas as pd

def parse_nhs_outpatient_data(url: str) -> tuple[pd.DataFrame, float, dict]:
    """
    Parses NHS England outpatient data from the official Excel workbook and returns:

    - A DataFrame with normalized female and male attendance percentages by age group.
    - The national first attendance ratio (first / total attendances).
    - A dictionary of appointment status proportions for 2023–24.

    Parameters
    ----------
    url : str
        Direct URL to the NHS Excel file (e.g., from https://digital.nhs.uk).

    Returns
    -------
    tuple
        DEFAULT_AGE_GENDER_PROBS : pd.DataFrame
            Columns: ['age_yrs', 'total_female', 'total_male'], values normalized to sum ≈ 1.
        DEFAULT_FIRST_ATTENDANCE_RATIO : float
            First attendances / total attendances (rounded to 5 decimals).
        DEFAULT_STATUS_RATES : dict
            Keys: {'attended', 'cancelled', 'did not attend', 'unknown'} with values ∈ [0, 1].
    """
    try:
        summary3 = pd.read_excel(url, sheet_name="Summary Report 3", header=5, nrows=20)
        summary2 = pd.read_excel(url, sheet_name="Summary Report 2", header=5, nrows=9)
        summary1 = pd.read_excel(url, sheet_name="Summary Report 1", header=5, nrows=12)
    except Exception as e:
        print(f"Error loading data from URL: {e}")
        return pd.DataFrame(), None, {}

    # --- Age and sex distribution (Summary Report 3) ---
    summary3.columns = [
        'age_yrs', 'attended_female_maternity', 'attended_female_non_maternity',
        'attended_male', 'dna_female', 'dna_male'
    ]
    summary3['attended_female'] = (
        summary3['attended_female_maternity'] + summary3['attended_female_non_maternity']
    )

    relevant_cols = ['attended_female', 'attended_male', 'dna_female', 'dna_male']
    total_count = summary3[relevant_cols].to_numpy().sum()

    if total_count == 0:
        return pd.DataFrame(), None, {}

    summary3[relevant_cols] /= total_count
    summary3['total_female'] = summary3['attended_female'] + summary3['dna_female']
    summary3['total_male'] = summary3['attended_male'] + summary3['dna_male']
    DEFAULT_AGE_GENDER_PROBS = summary3[['age_yrs', 'total_female', 'total_male']].round(5)

    # --- First attendance ratio (Summary Report 2) ---
    try:
        row = summary2[summary2.iloc[:, 0] == 'Total Activity'].squeeze()
        DEFAULT_FIRST_ATTENDANCE_RATIO = round(
            row['First Attendances'] / row['Attendances'], 5
        )
    except Exception as e:
        print(f"Error extracting first attendance ratio: {e}")
        DEFAULT_FIRST_ATTENDANCE_RATIO = None

    # --- Status distribution for 2023–24 (Summary Report 1) ---
    try:
        row = summary1[summary1['Year'] == '2023-24'].squeeze()
        DEFAULT_STATUS_RATES = {
            "attended": round(row['Attendances %'] / 100, 3),
            "did not attend": round(row['Did not attends (DNAs) %'] / 100, 3),
            "cancelled": round(
                (row['Patient cancellations %'] + row['Hospital cancellations %']) / 100, 3
            ),
            "unknown": round(row['Unknown %'] / 100, 3)
        }
    except Exception as e:
        print(f"Error extracting status rates: {e}")
        DEFAULT_STATUS_RATES = {}

    return DEFAULT_AGE_GENDER_PROBS, DEFAULT_FIRST_ATTENDANCE_RATIO, DEFAULT_STATUS_RATES


# -----------------------------------------------------------------------------
# Optional: To regenerate constants manually, uncomment and run the block below
# -----------------------------------------------------------------------------
# nhs_url = "https://files.digital.nhs.uk/34/18846B/hosp-epis-stat-outp-rep-tabs-2023-24-tab.xlsx"
# DEFAULT_AGE_GENDER_PROBS, DEFAULT_FIRST_ATTENDANCE_RATIO, DEFAULT_STATUS_RATES = parse_nhs_outpatient_data(nhs_url)