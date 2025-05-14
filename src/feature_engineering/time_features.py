import pandas as pd
from typing import Union, Optional


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