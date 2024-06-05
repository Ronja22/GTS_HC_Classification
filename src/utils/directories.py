import os

def create_directory_if_not_exists(path):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path)