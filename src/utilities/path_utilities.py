import os

# Global dictionary to keep track of directory changes
_directory_changes = {}

def ensure_parent_dir(flag_name='default_dir_change'):
    """
    Change the current working directory to its parent directory, but only once per session.
    
    This function is designed to be safely called multiple times, for example in notebook cells
    that might be re-executed. It will only change the directory the first time it's called
    with a given flag_name.
    
    Args:
        flag_name (str): A unique identifier for this specific directory change operation.
                         Use different names for different directory change operations.
                         
    Returns:
        str: The current working directory after the operation
    """
    global _directory_changes
    
    # Check if this specific directory change has already been executed
    if not _directory_changes.get(flag_name, False):
        # Get current directory
        original_dir = os.getcwd()
        print(f"Original directory: {original_dir}")
        
        # Change to parent directory
        parent_dir = os.path.dirname(original_dir)
        os.chdir(parent_dir)
        print(f"Changed directory to: {os.getcwd()}")
        
        # Mark this directory change as executed
        _directory_changes[flag_name] = True
    else:
        print(f"Directory already changed to: {os.getcwd()}")
    
    return os.getcwd()