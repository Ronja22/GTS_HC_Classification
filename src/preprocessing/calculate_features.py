import numpy as np
import pandas as pd
import os

import src.utils.directories as dir


def rotation_angles(rotation_matrix):
    """
    Calculate rotation angles (in degrees) from a given rotation matrix.

    Args:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: Tuple containing rotation angles around the X, Y, and Z axes (in degrees).
    """
    # Ensure the input is a 3x3 matrix
    
    if rotation_matrix.shape != (4, 4):
        raise ValueError("Input rotation_matrix must be a 3x3 matrix.")

    # Extract rotation angles using the appropriate arcsine and arctangent functions
    # Angle around the X-axis
    angle_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    # Angle around the Y-axis
    angle_y = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    # Angle around the Z-axis
    angle_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angles from radians to degrees
    angle_x_deg = np.degrees(angle_x)
    angle_y_deg = np.degrees(angle_y)
    angle_z_deg = np.degrees(angle_z)

    # Return the rotation angles as a tuple
    return angle_x_deg, angle_y_deg, angle_z_deg


def calc_features(coordinates, fps):
    """
    Calculate various movement features from given coordinates data at a specified frame rate.

    Args:
        coordinates (pd.DataFrame): DataFrame containing the coordinate data with a 'frame' column.
        fps (int): Frames per second of the data.

    Returns:
        tuple: Tuple containing DataFrames for each of the calculated features.
    """
# Create dictionaries to store lists of calculated features
    feature_data = {
        "negsum": {},
        "possum": {},
        "maxnegmov": {},
        "maxposmov": {},
        "maxnegmovpertime": {},
        "maxposmovpertime": {},
        "meanderneg": {},
        "meanderpos": {},
        "maxderneg": {},
        "maxderpos": {},
        "max_dist": {},
        "var": {}
    }
    # Iterate through each column in the coordinates DataFrame
    for col in coordinates.columns:
        if col != "frame":
            Cnegsum = []  # Sum of negative movements
            Cpossum = []  # Sum of positive movements
            Cmaxnegmov = []  # Maximum consecutive negative movement
            Cmaxposmov = []  # Maximum consecutive positive movement
            Cmaxnegmovpertime = []  # Max negative movement per time
            Cmaxposmovpertime = []  # Max positive movement per time
            Cmeanderneg = []  # Mean derivative of negative movements
            Cmeanderpos = []  # Mean derivative of positive movements
            Cmaxderneg = []  # Maximum negative derivative
            Cmaxderpos = []  # Maximum positive derivative
            Cmax_dist = []  # Maximum distance
            Cvar = []  # Variance

            # Iterate through each second of data
            for j in range(0, len(coordinates), fps):
                # Get the values for the current second
                values = np.asarray(coordinates[col].iloc[j:j + fps])
                if values.size > 0:
                    # Calculate frame-to-frame distances using numpy
                    dists = np.diff(values)

                    # Separate distances into negative and positive using numpy boolean indexing
                    neg_dists = dists[dists < 0]
                    pos_dists = dists[dists >= 0]

                    # Sum of negative distances
                    sum_neg_dists = np.sum(neg_dists)
                    # Sum of positive distances
                    sum_pos_dists = np.sum(pos_dists)

                    # Mean derivative of negative movements
                    mean_derivitive_neg = sum_neg_dists / (len(neg_dists) * (1 / fps)) if len(neg_dists) > 0 else 0
                    # Mean derivative of positive movements
                    mean_derivitive_pos = sum_pos_dists / (len(pos_dists) * (1 / fps)) if len(pos_dists) > 0 else 0

                    # Maximum negative derivative
                    maxdernegv = np.min(neg_dists) / fps if len(neg_dists) > 0 else 0
                    # Maximum positive derivative
                    maxderposv = np.max(pos_dists) / fps if len(pos_dists) > 0 else 0

                    # Maximum distance in the current second
                    maxdistv = np.ptp(values)  # np.ptp computes the range (max - min) of values

                    # Variance of values in the current second
                    varv = np.var(values)

                    # Calculate maximum consecutive positive movement and duration
                    if len(pos_dists) > 0 and sum(pos_dists) > 0:
                        max_sum = 0.0
                        max_length = 0
                        current_sum = 0
                        current_length = 0

                        for num in dists:
                            if num > 0:
                                # Accumulate positive movement
                                current_sum += num
                                # Increment length of current positive movement
                                current_length += 1
                            else:
                                # Check if current positive movement is greater than the current maximum
                                if current_sum > max_sum or (current_sum == max_sum and current_length < max_length):
                                    max_sum = current_sum
                                    max_length = current_length
                                # Reset current positive movement
                                current_sum = 0.0
                                current_length = 0

                        # Check if the last positive movement is greater than the current maximum
                        if current_sum > max_sum or (current_sum == max_sum and current_length < max_length):
                            max_sum = current_sum
                            max_length = current_length

                        # Calculate maximum positive movement per unit time
                        max_sum_per_time = max_sum / (max_length / fps)
                    else:
                        # If there are no positive movements, initialize values to zero
                        max_sum = 0
                        max_length = 0
                        max_sum_per_time = 0

                    # Calculate maximum consecutive negative movement and duration
                    if len(neg_dists) > 0:
                        min_sum = 0.0
                        min_length = 0
                        current_sum = 0
                        current_length = 0

                        for num in dists:
                            if num < 0:
                                # Accumulate negative movement
                                current_sum += num
                                # Increment length of current negative movement
                                current_length += 1
                            else:
                                # Check if current negative movement is smaller than the current minimum
                                if current_sum < min_sum or (current_sum == min_sum and current_length < min_length):
                                    min_sum = current_sum
                                    min_length = current_length
                                # Reset current negative movement
                                current_sum = 0.0
                                current_length = 0

                        # Check if the last negative movement is smaller than the current minimum
                        if current_sum < min_sum or (current_sum == min_sum and current_length < min_length):
                            min_sum = current_sum
                            min_length = current_length

                        # Calculate maximum negative movement per unit time
                        min_sum_per_time = min_sum / (min_length / fps)
                    else:
                        # If there are no negative movements, initialize values to zero
                        min_sum = 0
                        min_length = 0
                        min_sum_per_time = 0

                    # Append the calculated features to their respective lists
                    Cnegsum.append(sum_neg_dists)
                    Cpossum.append(sum_pos_dists)
                    Cmaxnegmov.append(min_sum)
                    Cmaxposmov.append(max_sum)
                    Cmaxnegmovpertime.append(min_sum_per_time)
                    Cmaxposmovpertime.append(max_sum_per_time)
                    Cmeanderneg.append(mean_derivitive_neg)
                    Cmeanderpos.append(mean_derivitive_pos)
                    Cmaxderneg.append(maxdernegv)
                    Cmaxderpos.append(maxderposv)
                    Cmax_dist.append(maxdistv)
                    Cvar.append(varv)
                else:
                    # Append zeros if no values are available
                    Cnegsum.append(0)
                    Cpossum.append(0)
                    Cmaxnegmov.append(0)
                    Cmaxposmov.append(0)
                    Cmaxnegmovpertime.append(0)
                    Cmaxposmovpertime.append(0)
                    Cmeanderneg.append(0)
                    Cmeanderpos.append(0)
                    Cmaxderneg.append(0)
                    Cmaxderpos.append(0)
                    Cmax_dist.append(0)
                    Cvar.append(0)

            # Assign lists to the respective dictionaries
            feature_data["negsum"][col] = Cnegsum
            feature_data["possum"][col] = Cpossum
            feature_data["maxnegmov"][col] = Cmaxnegmov
            feature_data["maxposmov"][col] = Cmaxposmov
            feature_data["maxnegmovpertime"][col] = Cmaxnegmovpertime
            feature_data["maxposmovpertime"][col] = Cmaxposmovpertime
            feature_data["meanderneg"][col] = Cmeanderneg
            feature_data["meanderpos"][col] = Cmeanderpos
            feature_data["maxderneg"][col] = Cmaxderneg
            feature_data["maxderpos"][col] = Cmaxderpos
            feature_data["max_dist"][col] = Cmax_dist
            feature_data["var"][col] = Cvar

    # Convert dictionaries of lists into DataFrames
    negsum = pd.DataFrame(feature_data["negsum"])
    possum = pd.DataFrame(feature_data["possum"])
    maxnegmov = pd.DataFrame(feature_data["maxnegmov"])
    maxposmov = pd.DataFrame(feature_data["maxposmov"])
    maxnegmovpertime = pd.DataFrame(feature_data["maxnegmovpertime"])
    maxposmovpertime = pd.DataFrame(feature_data["maxposmovpertime"])
    meanderneg = pd.DataFrame(feature_data["meanderneg"])
    meanderpos = pd.DataFrame(feature_data["meanderpos"])
    maxderneg = pd.DataFrame(feature_data["maxderneg"])
    maxderpos = pd.DataFrame(feature_data["maxderpos"])
    max_dist = pd.DataFrame(feature_data["max_dist"])
    var = pd.DataFrame(feature_data["var"])

    return negsum, possum, maxnegmov, maxposmov, maxnegmovpertime, maxposmovpertime, meanderneg, meanderpos, maxderneg, maxderpos, max_dist, var


def calc_all_features(folder, certain_subjects, override, fps, folder_extension = "filtered"):
    window_size = 3

    # Check if the directory for saving negative movement summaries exists
    checkpath = os.path.join(folder, folder_extension + "_Sum_neg_movements_X")
    
    # Loop through each file in the X coordinate folder
    for data_name in os.listdir(os.path.join(folder, "Coordinates_X_" + folder_extension)):
        if certain_subjects == [] or certain_subjects == None:
            certain_subjects = ["_"]
        if any(string in data_name for string in certain_subjects):
            # If override is False and the file already exists, skip processing
            if override == False:
                if os.path.exists(os.path.join(checkpath, data_name)):
                    print("file " + data_name + " exists, skipping")
                    continue
            print("data name", data_name)

            # Load X, Y, Z coordinates and transformation matrix
            X = pd.read_csv(os.path.join(folder, "Coordinates_X_" + folder_extension, data_name))
            Y = pd.read_csv(os.path.join(folder, "Coordinates_Y_" + folder_extension, data_name))
            Z = pd.read_csv(os.path.join(folder, "Coordinates_Z_" + folder_extension, data_name))
            T = np.load(os.path.join(folder, "Transformation_matrix_arrays/", data_name + ".npy"), allow_pickle=True)

            # TRANSLATION
            # Load original coordinates for translation calculation
            Xold = pd.read_csv(os.path.join(folder, "Coordinates_X_prepared", data_name))
            Yold = pd.read_csv(os.path.join(folder, "Coordinates_Y_prepared", data_name))
            Zold = pd.read_csv(os.path.join(folder, "Coordinates_Z_prepared", data_name))

            # Calculate translation and add as a new column
            X["translation"] = Xold["0"].rolling(window_size, center=True, min_periods=1).median()
            Y["translation"] = Yold["0"].rolling(window_size, center=True, min_periods=1).median()
            Z["translation"] = Zold["0"].rolling(window_size, center=True, min_periods=1).median()

            # Drop columns containing "Unnamed" in their name
            X = X.loc[:, ~X.columns.str.contains('Unnamed')]
            Y = Y.loc[:, ~Y.columns.str.contains('Unnamed')]
            Z = Z.loc[:, ~Z.columns.str.contains('Unnamed')]

            # ROTATION
            # Calculate and add rotation columns
            rotx = []
            roty = []
            rotz = []

            for i in range(0, len(T)):
                if type(T[i]) == float:
                    rotx.append(0)
                    roty.append(0)
                    rotz.append(0)
                else:
                    x, y, z = rotation_angles(T[i])
                    rotx.append(x)
                    roty.append(y)
                    rotz.append(z)

            X["Rot"] = pd.Series(rotx).rolling(window_size, center=True, min_periods=1).median()
            Y["Rot"] = pd.Series(roty).rolling(window_size, center=True, min_periods=1).median()
            Z["Rot"] = pd.Series(rotz).rolling(window_size, center=True, min_periods=1).median()

            # Ensure the length is 2.5 min for RUSH and HYPE, 
            # URGE videos are longer (5 min)
            frame_length = int(fps * 2.5 * 60)
            if "URGE" not in folder:
                print("cutting to 3750")
                X = X.iloc[:frame_length]
                Y = Y.iloc[:frame_length]
                Z = Z.iloc[:frame_length]

            # Create directories if they don't exist
            for feat in ["Sum_neg_movements", "Sum_pos_movements", "Max_neg_movements", "Max_pos_movements",
                         "Max_neg_mov_per_time", "Max_pos_mov_per_time", "Mean_derivative_neg", "Mean_derivative_pos",
                         "Max_derivative_neg", "Max_derivative_pos", "Max_dist", "Var"]:
                for ax in ["_X", "_Y", "_Z"]:
                    dir.create_directory_if_not_exists(os.path.join(folder, feat + ax))
            
        # Define a dictionary to store data for each axis
        axis_data = {"X": X, "Y": Y, "Z": Z}

        # Process each axis (X, Y, Z)
        for ax, data in axis_data.items():
            # Calculate features for the current axis
            negsum, possum, maxnegmov, maxposmov, maxnegmovpertime, maxposmovpertime, meanderneg, meanderpos, \
            maxderneg, maxderpos, max_dist, var = calc_features(data, fps)

            # Define the folder path for saving features
            feature_folder = os.path.join(folder, ax)

            dir.create_directory_if_not_exists(feature_folder)
            # Save calculated features to CSV files
            negsum.to_csv(os.path.join(feature_folder, "Sum_neg_movements.csv"))
            possum.to_csv(os.path.join(feature_folder, "Sum_pos_movements.csv"))
            maxnegmov.to_csv(os.path.join(feature_folder, "Max_neg_movements.csv"))
            maxposmov.to_csv(os.path.join(feature_folder, "Max_pos_movements.csv"))
            maxnegmovpertime.to_csv(os.path.join(feature_folder, "Max_neg_mov_per_time.csv"))
            maxposmovpertime.to_csv(os.path.join(feature_folder, "Max_pos_mov_per_time.csv"))
            meanderneg.to_csv(os.path.join(feature_folder, "Mean_derivative_neg.csv"))
            meanderpos.to_csv(os.path.join(feature_folder, "Mean_derivative_pos.csv"))
            maxderneg.to_csv(os.path.join(feature_folder, "Max_derivative_neg.csv"))
            maxderpos.to_csv(os.path.join(feature_folder, "Max_derivative_pos.csv"))
            var.to_csv(os.path.join(feature_folder, "Variance.csv"))
            max_dist.to_csv(os.path.join(feature_folder, "Max_distance.csv"))