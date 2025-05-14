import pandas as pd
import reverse_geocoder as rg
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