import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import src.utils.directories as dir

def transform_to_standardposition(folder_name, certain_subjects, override):
    """
    Apply transformation to bring data to a standard position based on transformation matrices.

    Args:
        folder_name (str): The path to the folder containing filtered data and transformation matrices.
        certain_subjects (list): List of specific subjects to transform.
        override (bool): If False, skip processing for existing files.

    Returns:
        None
    """
    for data_name in os.listdir(folder_name + "Coordinates_X_prepared"):
        if certain_subjects == []:
            certain_subjects = ["_"]

        # Check if the data name contains any of the specified subject identifiers
        if any(string in data_name for string in certain_subjects):
            print("Data name:", data_name)

            # If override is False, check if transformed coordinates already exist and skip this video
            if not override and os.path.exists(os.path.join(folder_name, "Coordinates_X_standard", data_name)):
                print("File " + data_name + " exists, skipping")
                continue

            # Load filtered data for X, Y, and Z coordinates
            X = pd.read_csv(os.path.join(folder_name, "Coordinates_X_prepared", data_name))
            Y = pd.read_csv(os.path.join(folder_name, "Coordinates_Y_prepared", data_name))
            Z = pd.read_csv(os.path.join(folder_name, "Coordinates_Z_prepared", data_name))

            # Determine the appropriate path for transformation matrices based on folder structure
            if "/URGE/" in folder_name:
                Tmat = pd.read_csv(os.path.join(folder_name, "transformation_cut", data_name))
            else:
                Tmat = pd.read_csv(os.path.join(folder_name, "Transformation_matrix", data_name))

            print("Transformation shape:", Tmat.shape[0])
            print("X shape:", X.shape[0])
            if X.shape[0] >4000:
                raise ValueError("Probably not downsampled")
                break

            # Adjust data frames if the number of rows doesn't match the transformation matrix
            if Tmat.shape[0] != X.shape[0]:
                X = X.drop(X.index[-1])
                Y = Y.drop(Y.index[-1])
                Z = Z.drop(Z.index[-1])
            print("Adjusted transformation shape:", Tmat.shape)
            print("Adjusted X shape:", X.shape)
            print("Adjusted Y shape:", Y.shape)
            print("Adjusted Z shape:", Z.shape)

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

            T = []

            # Iterate over data points and apply transformation
            for i in range(0, len(Y)):
                # Check if transmat is not NaN; if NaN, no transformation is applied
                if isinstance(Tmat["0"].iloc[i], (float, int)):
                    T.append(Tmat["0"].iloc[i])
                    print("+++nan++++")
                    print(Tmat["0"].iloc[i])
                else:
                    #print(i)
                    # Parse transformation matrix string and modify to standard position
                    transmat = parse_transmat_string(Tmat["0"].iloc[i])
                    transmat[:, 3] = [0, 0, 0, 1]

                    # Get the transformed data using dot product
                    #print(X.shape)
                    #print(Y.shape)
                    #print(Z.shape)
                    point_coords = np.vstack((X.iloc[i], Y.iloc[i], Z.iloc[i], np.ones_like(X.iloc[i])))
                    new_coords = np.dot(transmat, point_coords)

                    # Visualize transformed data for every 5000 points
                    if i % 5000 == 0:
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

            # Define save paths and save transformed data and matrices
            X_savepath = os.path.join(folder_name, "Coordinates_X_standard")
            Y_savepath = os.path.join(folder_name, "Coordinates_Y_standard")
            Z_savepath = os.path.join(folder_name, "Coordinates_Z_standard")
            T_savepath = os.path.join(folder_name, "Transformation_matrix_arrays")

            dir.create_directory_if_not_exists(X_savepath)
            X.to_csv(os.path.join(X_savepath, data_name))

            dir.create_directory_if_not_exists(Y_savepath)
            Y.to_csv(os.path.join(Y_savepath, data_name))

            dir.create_directory_if_not_exists(Z_savepath)
            Z.to_csv(os.path.join(Z_savepath, data_name))

            dir.create_directory_if_not_exists(T_savepath)
            np.save(os.path.join(T_savepath, data_name + ".npy"), T)