import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime as dt


'''
def calculate_date_difference(df):
    """
    Calculate the difference in months between 'last_date' and 'issue_daily'.
    """
    df['last_date_issue_difference'] = df.apply(
        lambda row: (relativedelta(row['last_date'], row['issue_daily']).years * 12) + 
                    relativedelta(row['last_date'], row['issue_daily']).months, 
        axis=1
    )
    return df
'''


def clean_term(df):
    """
    Replace 'term' string values by their numeric equivalents.
    """
    if 'term' in df.columns:
        df['term_clean'] = df['term'].replace({' 36 months': 36, ' 60 months': 60})
    else:
        print("The column 'term' does not exist in the DataFrame.")
    return df

def normalize_application_type(df):
    """
    Normalize 'application_type' to a consistent format and reduce categories.
    """
    if 'application_type' in df.columns:
        df['application_type'] = df['application_type'].str.lower()
        mapping = {
            'individual': ['individual', 'direct_pay'],
            'joint': ['joint app', 'joint'],
        }
        for category, values in mapping.items():
            df.loc[df['application_type'].isin(values), 'application_type'] = category
    else:
        print("The column 'application_type' does not exist in the DataFrame.")
    return df

def clean_revol_util(df):
    """
    Handle missing or incorrect values in 'revol_util' based on 'revol_bal'.
    """
    # Drop rows where 'revol_util' is NaN and 'revol_bal' > 0
    if 'revol_util' in df.columns:
        df = df.drop(df[(df['revol_util'].isna()) & (df['revol_bal'] > 0)].index)
        # Set 'revol_util' to 0 where 'revol_bal' is 0
        df.loc[df['revol_bal'] == 0, 'revol_util'] = 0
    else:  
        print("The column 'revol_util' does not exist in the DataFrame.")
    return df

def clean_employment_length(df):
    """
    Convert 'emp_length' to numeric, treating '<1 year' as 0.5 and '10+ years' as 10.
    """
    emp_length_map = {
        '10+ years': '10', '< 1 year': '0.5', '1 year': '1', '2 years': '2', '3 years': '3', '4 years': '4',
        '5 years': '5', '6 years': '6', '7 years': '7', '8 years': '8', '9 years': '9', 'n/a': np.nan
    }
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].replace(emp_length_map)
    else:  
        print("The column 'emp_length' does not exist in the DataFrame.")
    return df

def consolidate_home_ownership(df):
    """
    Group less common home ownership statuses into a single 'OTHER' category.
    """
    rare_categories = ['ANY', 'NONE', 'OTHER']
    if 'home_ownership' in df.columns:
        df['home_ownership'] = df['home_ownership'].replace(rare_categories, 'OTHER')
    else:
        print("The column 'home_ownership' does not exist in the DataFrame.")
    return df

def convert_issue_date_to_days_from_reference(df, date_column='issue_daily', reference_date='2024-05-15'):
    """
    Convert the 'issue_daily' column to a numerical feature that represents the number of days
    from the given date to a specified reference date (changeable in params). Errors in date parsing are coerced to NaT.

    Parameters:
        df (DataFrame): DataFrame containing date column.
        date_column (str): The name of the date column to convert, default is 'issue_daily'.
        reference_date (str or datetime): The reference date from which to calculate the days.

    Returns:
        DataFrame: DataFrame with the new numerical date column added and wihtout the initial datetime column.
    """
    # Parsing the reference date if it's provided as a string
    if 'issue_daily' in df.columns:
        if isinstance(reference_date, str):
            reference_date = dt.datetime.strptime(reference_date, '%Y-%m-%d')

        # Converting the date column to datetime format, it also handling any parsing errors.
        df['loan_date'] = pd.to_datetime(df[date_column], errors='coerce')

        # Calculating the difference in days from the reference date for the new datetime column
        df['loan_date_days_from_reference'] = df['loan_date'].apply(lambda x: (reference_date - x).days if pd.notna(x) else np.nan).astype('float64')

        # Droping the initial datetime column since no longer needed
        df.drop(columns=['issue_daily'], inplace=True)
    else:
        print("The column 'issue_daily' does not exist in the DataFrame.")
    return df

def info_col(df):
    column_info = {}

    # Populating the dictionary with data type and unique value count for each column
    for col in df.columns:
        column_info[col] = {
            'Data type': df[col].dtype,
            'Number of unique values': df[col].nunique(),
            'Missing values' : df[col].isnull().sum()/df.shape[0]*100
        }
    print('df shape:', df.shape)
    return pd.DataFrame.from_dict(column_info, orient='index').sort_values(by='Missing values', ascending=False)

def handle_missing_emp_length(df):
    """
    Fill missing values in 'emp_length' column with 0.
    """
    if 'emp_length' in df.columns:
        df.loc[df['emp_length'].isna(), 'emp_length'] = 0
    else:
        print("The column 'emp_length' does not exist in the DataFrame.")
    return df

def handle_missing_num_accts_ever_120_pd(df):
    """
    Fill missing values in 'num_tl_120dpd' column with 0.
    """
    if 'num_accts_ever_120_pd' in df.columns:
        df.loc[df['num_accts_ever_120_pd'].isna(), 'num_accts_ever_120_pd'] = 0
    else:
        print("The column 'num_accts_ever_120_pd' does not exist in the DataFrame.")
    return df

def fill_bc_util(df):
    """
    Fills missing values in the 'bc_util' column of the DataFrame with 0.

    Parameters:
    df (DataFrame): The pandas DataFrame to process.

    Returns:
    DataFrame: The DataFrame with 'bc_util' missing values filled.
    """
    # Check if 'bc_util' is in DataFrame
    if 'bc_util' in df.columns:
        df['bc_util'] = df['bc_util'].fillna(0)
    else:
        print("The column 'bc_util' does not exist in the DataFrame.")
    
    return df


def drop_missing_column(df, missing_threshold=10, axis=1):
    """
    Drops columns or rows from the DataFrame based on a missing data threshold.

    Parameters:
    df (DataFrame): The pandas DataFrame to process.
    missing_threshold (float): The maximum allowed percentage of missing values.
    axis (int): Axis on which to operate (0 for rows, 1 for columns).

    Returns:
    DataFrame: DataFrame with rows/columns dropped based on the missing data threshold.
    """
    # Calculate the threshold for non-missing values
    # Convert missing_threshold to a fraction (e.g., 10% -> 0.9)
    threshold = int((100 - missing_threshold) / 100.0 * df.shape[1-axis])
    
    # Drop rows or columns with missing data exceeding the threshold
    df_cleaned = df.dropna(axis=axis, thresh=threshold)
    
    return df_cleaned

def drop_missing_data(df):
    """
    Drop rows or columns from the DataFrame based on the missing data threshold.

    Parameters:
    df (DataFrame): The pandas DataFrame to process.

    Returns:
    DataFrame: DataFrame with rows/columns dropped based on the missing data threshold.
    """
    # Drop rows with missing data
    df_cleaned = df.dropna()
    
    return df_cleaned

def clean_data(df):
    """
    Apply all cleaning steps to the dataframe.
    """
    #df = calculate_date_difference(df)

    df = clean_term(df)
    df = normalize_application_type(df)
    df = clean_revol_util(df)
    df = clean_employment_length(df)
    df = consolidate_home_ownership(df)
    df = convert_issue_date_to_days_from_reference(df)
    df = handle_missing_emp_length(df)
    df = handle_missing_num_accts_ever_120_pd(df)
    df = drop_missing_column(df)
    df = drop_missing_data(df)

    print('Called functions: \n\
        - clean_term (term_clean)\n\
        - normalize_application_type\n\
        - clean_revol_util\n\
        - clean_employment_length\n\
        - consolidate_home_ownership\n\
        - convert_issue_date_to_days_from_reference\n\
        - handle_missing_emp_length\n\
        - handle_missing_num_accts_ever_120_pd\n\
        - fill_bc_util\n\
        - drop_missing_column\n\
        - drop_missing_data\n')
      
    print('Dataframe cleaned, Dataframe shape:', df.shape)
    return df
