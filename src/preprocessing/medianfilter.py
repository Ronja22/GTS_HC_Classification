import os
import pandas as pd
import numpy as np 

import src.utils.directories as dir

def medianfilter(folder, certain_subjects=[], window_size=3, override=False):
    """
    Apply column-wise median filter to all files in 'Coordinates_X_prepared' (Y, Z) folders.

    Args:
        folder (str): Path to the main folder containing 'Coordinates_X_prepared' subfolders.
        certain_subjects (list, optional): List of specific subjects to filter. Default is an empty list.
        window_size (int, optional): Size of the rolling window for median filtering. Default is 3.
        override (bool, optional): If False, skip processing for existing files. Default is False.

    Returns:
        None
    """

    # Loop over X, Y, and Z axes
    for ax in ["X", "Y", "Z"]:
        folder_name = os.path.join(folder, "Coordinates_" + ax + "_centered_normalized")
        print("Data will be taken from:", folder_name)
        
        savepath = os.path.join(folder, "Coordinates_" + ax + "_filtered")
        print("Filtered data will be saved here:", savepath)
        
        dir.create_directory_if_not_exists(savepath)  # Create save path if it does not exist

        # Loop through files in the specified folder
        for data_name in os.listdir(folder_name):
            if not certain_subjects:
                certain_subjects = ["_"]

            # Check if the data name contains any of the specified subject identifiers
            if any(subject in data_name for subject in certain_subjects):

                # If override is False, skip files that already exist
                if not override and os.path.exists(os.path.join(savepath, data_name)):
                    print("File " + data_name + " exists, skipping")
                    continue

                print("Processing filename:", data_name)

                # Load prepared coordinates from CSV
                coordinates = pd.read_csv(os.path.join(folder_name, data_name), index_col=None)

                # Remove 'Unnamed: 0' column if it exists
                coordinates = coordinates.loc[:, ~coordinates.columns.str.contains('^Unnamed')]

                # Identify columns to apply median filter to (excluding 'frame' column)
                cols_to_filter = [col for col in coordinates.columns if col != 'frame']

                # Apply median filter to the specified columns
                coordinates[cols_to_filter] = coordinates[cols_to_filter].rolling(window_size, center=True, min_periods=1).median()

                # Save the filtered coordinates to a new CSV file
                coordinates.to_csv(os.path.join(savepath, data_name), index=False)
