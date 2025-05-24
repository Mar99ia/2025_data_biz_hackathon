import pandas as pd
import json

def read_inpost_data(file_path='data/raw/paczkomaty.json', method='pandas', display_info=True):
    """
    Read InPost data from JSON file and return as pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON file (default: 'data/raw/paczkomaty.json')
    method : str
        Method to read the file ('pandas' or 'json', default: 'pandas')
    display_info : bool
        Whether to display basic dataset information (default: True)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the InPost data
    """
    
    if method == 'pandas':
        # Method 1: Direct pandas read_json (simplest)
        df = pd.read_json(file_path)
    elif method == 'json':
        # Method 2: Using json library first (more control)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            df = pd.DataFrame(data)
    else:
        raise ValueError("Method must be either 'pandas' or 'json'")
    
    if display_info:
        # Display basic info about the dataset
        print("Dataset shape:", df.shape)
        print("\nColumn names:")
        print(df.columns.tolist())
        
        # Display first few rows
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        # Check data types
        print("\nData types:")
        print(df.dtypes)
    
    return df

# Example usage:
# df = read_inpost_data()  # Uses default settings
# df = read_inpost_data(method='json', display_info=False)  # Using json method without info display
# df = read_inpost_data('path/to/your/file.json')  # Custom file path