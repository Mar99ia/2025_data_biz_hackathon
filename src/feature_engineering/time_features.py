import pandas as pd
from typing import Union, Optional
import datetime


def calculate_time_since_last_transaction(
    df: pd.DataFrame, 
    user_id_col: str = 'user_id',
    timestamp_col: str = 'timestamp',
    new_col_name: str = 'time_since_last_txn',
    default_value: Union[int, float] = -9
) -> pd.DataFrame:
    """
    Calculate the time in seconds since a user's last transaction.
    
    This function sorts the transactions by user and timestamp, then calculates
    the time difference between consecutive transactions for each user.
    The first transaction for each user will get the specified default value.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing transaction data
    user_id_col : str, default='user_id'
        The name of the column containing user identifiers
    timestamp_col : str, default='timestamp'
        The name of the column containing transaction timestamps
    new_col_name : str, default='time_since_last_txn'
        The name to give to the new column
    default_value : Union[int, float], default=-9
        The value to use for the first transaction of each user
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new column added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'user_id': ['U1', 'U1', 'U2', 'U2'],
    ...     'timestamp': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-01', '2022-01-03'])
    ... })
    >>> calculate_time_since_last_transaction(df)
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Sort by user and timestamp
    result_df = result_df.sort_values([user_id_col, timestamp_col])
    
    # Calculate time difference in seconds
    result_df[new_col_name] = result_df.groupby(user_id_col)[timestamp_col].diff().dt.total_seconds()
    
    # Fill NaN values (first transaction for each user)
    result_df[new_col_name] = result_df[new_col_name].fillna(default_value)
    
    return result_df


def extract_hour_from_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    new_col_name: str = 'hour_of_day',
    convert_to_categorical: bool = False
) -> pd.DataFrame:
    """
    Extract the hour (0-23) from a timestamp column.
    
    This function extracts the hour component from a timestamp column and
    adds it as a new feature. The hour can optionally be converted to a 
    categorical data type.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing transaction data
    timestamp_col : str, default='timestamp'
        The name of the column containing transaction timestamps
    new_col_name : str, default='hour_of_day'
        The name to give to the new column
    convert_to_categorical : bool, default=False
        Whether to convert the hour to a categorical data type
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new hour column added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.to_datetime(['2022-01-01 08:30:00', '2022-01-01 15:45:00', '2022-01-02 22:15:00'])
    ... })
    >>> extract_hour_from_timestamp(df)
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Extract hour from timestamp
    result_df[new_col_name] = pd.to_datetime(result_df[timestamp_col]).dt.hour
    
    # Convert to categorical if requested
    if convert_to_categorical:
        result_df[new_col_name] = result_df[new_col_name].astype('category')
    
    return result_df


def extract_weekday_from_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    new_col_name: str = 'day_of_week',
    convert_to_categorical: bool = False,
    as_name: bool = False
) -> pd.DataFrame:
    """
    Extract the day of the week (0-6, where 0 is Monday) from a timestamp column.
    
    This function extracts the weekday component from a timestamp column and
    adds it as a new feature. The weekday can optionally be converted to a
    categorical data type or represented as names (Monday, Tuesday, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing transaction data
    timestamp_col : str, default='timestamp'
        The name of the column containing transaction timestamps
    new_col_name : str, default='day_of_week'
        The name to give to the new column
    convert_to_categorical : bool, default=False
        Whether to convert the weekday to a categorical data type
    as_name : bool, default=False
        Whether to use weekday names (Monday, Tuesday, etc.) instead of integers
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new weekday column added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.to_datetime(['2022-05-09', '2022-05-10', '2022-05-11'])
    ... })
    >>> extract_weekday_from_timestamp(df)  # Returns 0, 1, 2 (Monday, Tuesday, Wednesday)
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Extract weekday from timestamp (0 = Monday, 6 = Sunday)
    result_df[new_col_name] = pd.to_datetime(result_df[timestamp_col]).dt.weekday
    
    # Convert to weekday names if requested
    if as_name:
        weekday_names = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        result_df[new_col_name] = result_df[new_col_name].map(weekday_names)
    
    # Convert to categorical if requested
    if convert_to_categorical:
        result_df[new_col_name] = result_df[new_col_name].astype('category')
    
    return result_df


def extract_date_from_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    new_col_name: str = 'date'
) -> pd.DataFrame:
    """
    Extract the date component from a timestamp column.
    
    This function extracts just the date part (year-month-day) from a timestamp
    column, removing the time component. This is useful for grouping transactions
    by date or analyzing patterns on a daily basis.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing transaction data
    timestamp_col : str, default='timestamp'
        The name of the column containing transaction timestamps
    new_col_name : str, default='date'
        The name to give to the new column containing the date component
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new date column added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.to_datetime(['2022-01-01 08:30:00', '2022-01-02 15:45:00'])
    ... })
    >>> extract_date_from_timestamp(df)  # Returns '2022-01-01', '2022-01-02'
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Extract just the date component from the timestamp
    result_df[new_col_name] = pd.to_datetime(result_df[timestamp_col]).dt.date
    
    return result_df


def extract_week_of_year(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    new_col_name: str = 'week_of_year',
    convert_to_categorical: bool = False
) -> pd.DataFrame:
    """
    Extract the week number of the year (1-53) from a timestamp column.
    
    This function extracts the ISO week number from a timestamp column, which
    can be useful for detecting seasonal patterns in transactions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing transaction data
    timestamp_col : str, default='timestamp'
        The name of the column containing transaction timestamps
    new_col_name : str, default='week_of_year'
        The name to give to the new column
    convert_to_categorical : bool, default=False
        Whether to convert the week number to a categorical data type
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new week of year column added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.to_datetime(['2022-01-01', '2022-06-15', '2022-12-31'])
    ... })
    >>> extract_week_of_year(df)  # Returns week numbers: 52, 24, 52
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Extract ISO week number from timestamp (1-53)
    result_df[new_col_name] = pd.to_datetime(result_df[timestamp_col]).dt.isocalendar().week
    
    # Convert to categorical if requested
    if convert_to_categorical:
        result_df[new_col_name] = result_df[new_col_name].astype('category')
    
    return result_df


def convert_date_to_unix_timestamp(
    df: pd.DataFrame,
    date_col: str = 'date',
    new_col_name: str = 'unix_timestamp'
) -> pd.DataFrame:
    """
    Convert date objects to Unix timestamps (seconds since January 1, 1970).
    
    This function converts date values to Unix timestamps, which can be
    useful for certain machine learning models that work better with numeric inputs.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing date data
    date_col : str, default='date'
        The name of the column containing date values
    new_col_name : str, default='unix_timestamp'
        The name to give to the new column
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new unix timestamp column added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.to_datetime(['2022-01-01', '2022-01-02']).dt.date
    ... })
    >>> convert_date_to_unix_timestamp(df)
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Define a function to convert date to unix timestamp
    def date_to_unix(date_obj):
        if pd.isnull(date_obj):
            return None
        if isinstance(date_obj, datetime.date):
            # Convert datetime.date to unix timestamp (seconds since epoch)
            return int(datetime.datetime.combine(date_obj, datetime.time()).timestamp())
        return None
    
    # Apply the conversion function
    result_df[new_col_name] = result_df[date_col].apply(date_to_unix)
    
    return result_df