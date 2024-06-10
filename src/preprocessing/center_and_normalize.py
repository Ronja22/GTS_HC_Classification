import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import src.utils.directories as dir


def center_and_normalize_data(folder, certain_subjects, override, center=True, debug = False):
    """
    Center and normalize face mesh data.

    Args:
        folder (str): The path to the main folder containing the standardized face mesh data.
        certain_subjects (list): List of specific subjects to process.
        override (bool): If False, skip processing for existing files.
        center (bool, optional): If True, center the data before normalization. Default is True.

    Returns:
        None
    """

    # Determine save folders based on whether centering is applied
    if center:
        savefolder_X = os.path.join(folder, "Coordinates_X_centered_normalized")
        savefolder_Y = os.path.join(folder, "Coordinates_Y_centered_normalized")
        savefolder_Z = os.path.join(folder, "Coordinates_Z_centered_normalized")
    else:
        savefolder_X = os.path.join(folder, "Coordinates_X_normalized")
        savefolder_Y = os.path.join(folder, "Coordinates_Y_normalized")
        savefolder_Z = os.path.join(folder, "Coordinates_Z_normalized")

    # Iterate through all files in the standardized coordinates folder
    for filename in os.listdir(os.path.join(folder, "Coordinates_X_standard")):
        if not certain_subjects:
            certain_subjects = ["_"]

        # Process files that match any of the certain_subjects strings
        if any(subject in filename for subject in certain_subjects):
            print("Processing filename:", filename)

            # If override is False, skip files that already exist in the save folder
            if not override and os.path.exists(os.path.join(savefolder_X, filename)):
                print("File " + filename + " already exists, skipping")
                continue

            # Load standardized data for X, Y, and Z coordinates
            X = pd.read_csv(os.path.join(folder, "Coordinates_X_standard", filename))
            Y = pd.read_csv(os.path.join(folder, "Coordinates_Y_standard", filename))
            Z = pd.read_csv(os.path.join(folder, "Coordinates_Z_standard", filename))

            # Center the data if the center flag is set
            if center:
                X = subtract_one_point(X)
                Y = subtract_one_point(Y)
                Z = subtract_one_point(Z)

            # Normalize the face mesh data
            XX, YY, ZZ = normalize_face_mesh(X, Y, Z)

            if debug:
                # Plot a sample of the normalized data for visualization
                for i in range(0, 600, 100):
                    try:
                        plt.scatter(XX.iloc[i], -YY.iloc[i], s=0.2)
                    except:
                        pass
                plt.show()

            # Ensure save directories exist
            dir.create_directory_if_not_exists(savefolder_X)
            dir.create_directory_if_not_exists(savefolder_Y)
            dir.create_directory_if_not_exists(savefolder_Z)

            # Save the normalized data
            XX.to_csv(os.path.join(savefolder_X, filename), index=False)
            YY.to_csv(os.path.join(savefolder_Y, filename), index=False)
            ZZ.to_csv(os.path.join(savefolder_Z, filename), index=False)


def subtract_one_point(coordinates):
    """
    Subtract the coordinates of the first point (assumed to be the middle of the upper lip) from all points.

    Args:
        coordinates (pd.DataFrame): DataFrame containing the face mesh coordinates with a 'frame' column.

    Returns:
        pd.DataFrame: DataFrame with coordinates centered around the first point.
    """
    # Create a copy of the coordinates DataFrame
    coordinates_copy = coordinates.copy()

    # Subtract the coordinates of the first point from all points
    coordinates_copy = coordinates_copy.sub(coordinates.iloc[:, 0], axis=0)

    # Restore the 'frame' column
    coordinates_copy["frame"] = coordinates["frame"]

    return coordinates_copy

def normalize_face_mesh(X, Y, Z):
    """
    Normalize the face mesh coordinates by computing the mean and standard deviation of each frame.

    Args:
        X (pd.DataFrame): DataFrame containing the X coordinates of the face mesh.
        Y (pd.DataFrame): DataFrame containing the Y coordinates of the face mesh.
        Z (pd.DataFrame): DataFrame containing the Z coordinates of the face mesh.

    Returns:
        tuple: Normalized X, Y, and Z DataFrames.
    """
    # Initialize lists to hold the normalized data
    normalized_data = []

    # Iterate over each axis (X, Y, Z) for normalization
    for ax in [X, Y, Z]:
        # Remove columns containing 'Unnamed' or 'frame'
        ax = ax.loc[:, ~ax.columns.str.contains('Unnamed', case=False)]
        ax = ax.loc[:, ~ax.columns.str.contains('frame', case=False)]

        # Calculate the standard deviation and mean of each frame
        ax = ax.assign(std=ax.std(axis=1))
        ax = ax.assign(mean=ax.mean(axis=1))

        # Calculate the mean of means and mean of standard deviations for non-zero frames
        m = np.mean(ax[ax.sum(axis=1) != 0]["mean"])
        std1 = np.mean(ax[ax.sum(axis=1) != 0]["std"])

        # Drop the 'std' and 'mean' columns
        ax = ax.drop(columns=["std", "mean"])

        # Normalize the coordinates
        ax[ax.sum(axis=1) != 0] = ax[ax.sum(axis=1) != 0].applymap(lambda x: (x - m) / std1)

        # Append the normalized DataFrame to the list
        normalized_data.append(ax)

    # Return the normalized X, Y, and Z DataFrames
    return tuple(normalized_data)