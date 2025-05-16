import pandas as pd
import reverse_geocoder as rg
import pycountry
from typing import Dict, List, Union, Tuple, Optional


def _process_single_location(location: Dict) -> Tuple[float, float]:
    """
    Extract latitude and longitude from a location dictionary.
    
    Parameters
    ----------
    location : Dict
        Dictionary containing location data with 'lat' and 'long' keys
        
    Returns
    -------
    Tuple[float, float]
        A tuple containing (latitude, longitude)
    """
    if isinstance(location, dict) and 'lat' in location and 'long' in location:
        return (location['lat'], location['long'])
    return (None, None)


def extract_location_info(
    df: pd.DataFrame,
    location_column: str = 'location',
    add_country: bool = True,
    add_city: bool = True,
    add_admin_regions: bool = True
) -> pd.DataFrame:
    """
    Extract country, city, and administrative region information from location coordinates.
    
    This function extracts latitude and longitude values from a location column,
    then uses reverse geocoding to identify the country, city, and administrative regions 
    associated with those coordinates. Uses pandas apply for faster processing.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing location data
    location_column : str, default='location'
        The name of the column containing location data (formatted as a dict with 'lat' and 'long' keys)
    add_country : bool, default=True
        Whether to add a 'country' column to the dataframe
    add_city : bool, default=True
        Whether to add a 'city' column to the dataframe
    add_admin_regions : bool, default=True
        Whether to add administrative region columns ('administrative_region_1' and 'administrative_region_2')
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new columns added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'location': [
    ...         {'lat': 51.5214588, 'long': -0.1729636},
    ...         {'lat': 9.936033, 'long': 76.259952},
    ...         {'lat': 37.38605, 'long': -122.08385}
    ...     ]
    ... })
    >>> extract_location_info(df)
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Extract coordinates using apply (vectorized operation)
    coordinates = result_df[location_column].apply(_process_single_location)
    
    # Create a list of valid coordinates for geocoding
    valid_coords = []
    valid_indices = []
    
    for i, coord in enumerate(coordinates):
        if coord[0] is not None and coord[1] is not None:
            valid_coords.append(coord)
            valid_indices.append(i)
    
    # If we have valid coordinates, perform reverse geocoding
    if valid_coords:
        # Perform reverse geocoding for valid coordinates
        geocode_results = rg.search(valid_coords)
        
        # Prepare dictionaries for results
        country_dict = {}
        city_dict = {}
        admin1_dict = {}
        admin2_dict = {}
        
        # Process results
        for i, idx in enumerate(valid_indices):
            result = geocode_results[i]
            country_dict[idx] = result.get('cc', '')
            city_dict[idx] = result.get('name', '')
            admin1_dict[idx] = result.get('admin1', '')
            admin2_dict[idx] = result.get('admin2', '')
        
        # Add columns to the dataframe using vectorized operations
        if add_country:
            result_df['country'] = pd.Series([country_dict.get(i, '') for i in range(len(result_df))])
        
        if add_city:
            result_df['city'] = pd.Series([city_dict.get(i, '') for i in range(len(result_df))])
        
        if add_admin_regions:
            result_df['administrative_region_1'] = pd.Series([admin1_dict.get(i, '') for i in range(len(result_df))])
            result_df['administrative_region_2'] = pd.Series([admin2_dict.get(i, '') for i in range(len(result_df))])
    
    return result_df


def extract_location_info_batch(
    df: pd.DataFrame,
    location_column: str = 'location',
    batch_size: int = 1000,
    add_country: bool = True,
    add_city: bool = True,
    add_admin_regions: bool = True
) -> pd.DataFrame:
    """
    Extract country, city, and administrative region information from location coordinates using batching for very large datasets.
    
    This function processes the dataframe in batches to avoid memory issues with large datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing location data
    location_column : str, default='location'
        The name of the column containing location data (formatted as a dict with 'lat' and 'long' keys)
    batch_size : int, default=1000
        The number of records to process in each batch
    add_country : bool, default=True
        Whether to add a 'country' column to the dataframe
    add_city : bool, default=True
        Whether to add a 'city' column to the dataframe
    add_admin_regions : bool, default=True
        Whether to add administrative region columns ('administrative_region_1' and 'administrative_region_2')
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new columns added
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Create empty columns for results
    if add_country:
        result_df['country'] = ''
    if add_city:
        result_df['city'] = ''
    if add_admin_regions:
        result_df['administrative_region_1'] = ''
        result_df['administrative_region_2'] = ''
    
    # Process the dataframe in batches
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = result_df.iloc[start_idx:end_idx].copy()
        
        batch_result = extract_location_info(
            batch_df,
            location_column=location_column,
            add_country=add_country,
            add_city=add_city,
            add_admin_regions=add_admin_regions
        )
        
        # Update the results
        if add_country:
            result_df.loc[start_idx:end_idx-1, 'country'] = batch_result['country'].values
        if add_city:
            result_df.loc[start_idx:end_idx-1, 'city'] = batch_result['city'].values
        if add_admin_regions:
            result_df.loc[start_idx:end_idx-1, 'administrative_region_1'] = batch_result['administrative_region_1'].values
            result_df.loc[start_idx:end_idx-1, 'administrative_region_2'] = batch_result['administrative_region_2'].values
    
    return result_df


def add_country_match_feature(
    df: pd.DataFrame,
    user_country_col: str = 'user_country',
    transaction_country_col: str = 'country',
    new_col_name: str = 'is_user_transaction_country_match'
) -> pd.DataFrame:
    """
    Add a binary feature indicating if user country matches transaction country.
    
    This function compares the user's country with the country derived from 
    the transaction's geo-coordinates and adds a binary indicator column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing user country and transaction country data
    user_country_col : str, default='user_country'
        The name of the column containing user country information
    transaction_country_col : str, default='country'
        The name of the column containing transaction country information 
        (typically derived from geo-coordinates)
    new_col_name : str, default='is_user_transaction_country_match'
        The name to give to the new binary feature column
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new binary feature column added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'user_country': ['US', 'UK', 'FR', 'DE'],
    ...     'country': ['US', 'FR', 'FR', 'IT']
    ... })
    >>> add_country_match_feature(df)
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Compare user country with transaction country
    # Convert to integers (0/1) for consistency with other features
    result_df[new_col_name] = (
        result_df[user_country_col] == result_df[transaction_country_col]
    ).astype(int)
    
    return result_df


def convert_country_codes_to_names(
    df: pd.DataFrame,
    country_col: str = 'country',
    new_col_name: str = 'country_name',
    keep_original: bool = True
) -> pd.DataFrame:
    """
    Convert ISO 3166-1 alpha-2 country codes to full country names.
    
    This function takes a column containing ISO 3166-1 alpha-2 country codes
    and adds a new column with the full country names using the pycountry library.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing country code data
    country_col : str, default='country'
        The name of the column containing country codes
    new_col_name : str, default='country_name'
        The name to give to the new column containing full country names
    keep_original : bool, default=True
        Whether to keep the original country code column
        
    Returns
    -------
    pd.DataFrame
        A dataframe with the new country name column added
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'country': ['US', 'GB', 'FR', 'DE']
    ... })
    >>> convert_country_codes_to_names(df)
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Create a function to get the country name from a country code
    def get_country_name(code):
        if not code or not isinstance(code, str) or len(code) != 2:
            return None
        try:
            country = pycountry.countries.get(alpha_2=code)
            return country.name if country else None
        except (AttributeError, KeyError, ValueError):
            return None
    
    # Apply the function to the country code column using vectorized operations
    result_df[new_col_name] = result_df[country_col].apply(get_country_name)
    
    # Drop the original column if requested
    if not keep_original:
        result_df = result_df.drop(columns=[country_col])
    
    return result_df