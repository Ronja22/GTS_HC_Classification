import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import src.utils.directories as dir


def parse_transmat_string(string):
    """
    Parses a string representing a transformation matrix into a 4x4 NumPy array.

    Args:
        string (str): The input string containing the transformation matrix.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the parsed transformation matrix.

    Raises:
        ValueError: If the input string does not contain exactly 16 numeric elements.
    """
    # Preprocess the string to remove square brackets and split into lines
    lines = string.strip().replace('[', '').replace(']', '').split('\n')

    # Extract numeric values from each line, ignoring any non-numeric values
    numeric_elements = []
    for line in lines:
        elements = line.split()
        for e in elements:
            try:
                numeric_elements.append(float(e))
            except ValueError:
                continue  # Skip non-numeric elements

    # Validate the number of elements
    if len(numeric_elements) != 16:
        raise ValueError("Input string does not contain 16 numeric elements.")

    # Reshape elements into a 4x4 array
    array = np.array(numeric_elements).reshape(4, 4)

    return array



def transform_to_standardposition(folder_name, certain_subjects, override,debug = False):
    """
    Apply transformation to bring data to a standard position based on transformation matrices.

    Args:
        folder_name (str): The path to the folder containing filtered data and transformation matrices.
        certain_subjects (list): List of specific subjects to transform.
        override (bool): If False, skip processing for existing files.

    Returns:
        None
    """
    # Process each data file in the Coordinates_X_prepared directory
    for data_name in os.listdir(os.path.join(folder_name, "Coordinates_X_prepared")):
        if not certain_subjects:
            certain_subjects = ["_"]

        # Check if the data name contains any of the specified subject identifiers
        if any(subject in data_name for subject in certain_subjects):
            print("Processing data file:", data_name)

            # If override is False, skip files that already exist in the Coordinates_X_standard directory
            if not override and os.path.exists(os.path.join(folder_name, "Coordinates_X_standard", data_name)):
                print("File " + data_name + " already exists, skipping")
                continue

            # Load filtered data for X, Y, and Z coordinates
            X = pd.read_csv(os.path.join(folder_name, "Coordinates_X_prepared", data_name))
            Y = pd.read_csv(os.path.join(folder_name, "Coordinates_Y_prepared", data_name))
            Z = pd.read_csv(os.path.join(folder_name, "Coordinates_Z_prepared", data_name))

            # Determine the appropriate path for transformation matrices based on folder structure
            if "/URGE/" in folder_name:
                Tmat = pd.read_csv(os.path.join(folder_name, "Transformation_cut", data_name))
            else:
                Tmat = pd.read_csv(os.path.join(folder_name, "Transformation_matrix", data_name))

            print("Transformation matrix shape:", Tmat.shape[0])
            print("X shape:", X.shape[0])

            # Check if the data is downsampled; raise an error if it has more than 4000 frames
            if X.shape[0] > 4000:
                raise ValueError("Data appears to be not downsampled, aborting.")
                break
            print(Tmat.shape)
            print(X.shape[0])
            # Adjust data frames if the number of rows doesn't match the transformation matrix
            if Tmat.shape[0] != X.shape[0]:
                X = X.drop(X.index[-1])
                Y = Y.drop(Y.index[-1])
                Z = Z.drop(Z.index[-1])
                print("Adjusted transformation matrix and data shapes")

            # Extract frame information and drop unnecessary columns
            xframe = X["frame"]
            yframe = Y["frame"]
            zframe = Z["frame"]
            try:
                X = X.drop(columns=["Unnamed: 0"])
                Y = Y.drop(columns=["Unnamed: 0"])
                Z = Z.drop(columns=["Unnamed: 0"])
            except KeyError:
                pass  # Columns are not present, so no action is taken
            X = X.drop(columns=["frame"])
            Y = Y.drop(columns=["frame"])
            Z = Z.drop(columns=["frame"])

            # Initialize list to store transformed matrices
            T = []

            # Iterate over each frame to apply the transformation
            for i in range(len(Y)):
                # Check if the transformation matrix is NaN; if NaN, skip transformation
                if isinstance(Tmat.iloc[i, 0], (float, int)):
                    T.append(Tmat.iloc[i, 0])
        
                else:
                    # Parse transformation matrix string and set the translation component to zero
                    transmat = parse_transmat_string(Tmat.iloc[i, 0])
                    transmat[:, 3] = [0, 0, 0, 1]
                
                    
                    # Combine X, Y, and Z coordinates and apply transformation
                    point_coords = np.vstack((X.iloc[i], Y.iloc[i], Z.iloc[i], np.ones(X.shape[1])))
                    new_coords = np.dot(transmat, point_coords)

                    # Visualize transformed data for every 5000 points
                    if i % 5000 == 0 and debug:
                        plt.scatter(point_coords[0], -point_coords[1], c="b")
                        plt.show()
                        plt.scatter(new_coords[0], -new_coords[1], c="r")
                        plt.show()

                    T.append(transmat)
                    X.iloc[i] = new_coords[0]
                    Y.iloc[i] = new_coords[1]
                    Z.iloc[i] = new_coords[2]

            # Restore frame information and save transformed data and matrices
            X["frame"] = xframe
            Y["frame"] = yframe
            Z["frame"] = zframe

            print("Final X shape:", X.shape)
            print("Final Y shape:", Y.shape)
            print("Final Z shape:", Z.shape)
            print("Final transformation shape:", len(T))

            # Define save paths and ensure directories exist
            X_savepath = os.path.join(folder_name, "Coordinates_X_standard")
            Y_savepath = os.path.join(folder_name, "Coordinates_Y_standard")
            Z_savepath = os.path.join(folder_name, "Coordinates_Z_standard")
            T_savepath = os.path.join(folder_name, "Transformation_matrix_arrays")

            dir.create_directory_if_not_exists(X_savepath)
            X.to_csv(os.path.join(X_savepath, data_name), index=False)

            dir.create_directory_if_not_exists(Y_savepath)
            Y.to_csv(os.path.join(Y_savepath, data_name), index=False)

            dir.create_directory_if_not_exists(Z_savepath)
            Z.to_csv(os.path.join(Z_savepath, data_name), index=False)

            dir.create_directory_if_not_exists(T_savepath)
            np.save(os.path.join(T_savepath, data_name + ".npy"), T)