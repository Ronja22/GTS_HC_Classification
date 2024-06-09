import os
import pandas as pd
import helper_functions as hf  
import re
import numpy as np
import matplotlib.pyplot as plt

def medianfilter(folder, certain_subjects=[], window_size=3, override=False):
    """
    Apply column-wise median filter to all files in 'Coordinates_X_prepared' (Y,Z) folders.

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
        if "/URGE/" in folder:
            folder_name = os.path.join(folder, "Coordinates_" + ax + "_centered_normalized")
        else:
            folder_name = os.path.join(folder, "Coordinates_" + ax + "_centered_normalized")
        print("data will be taken from", folder_name)
        
        savepath = os.path.join(folder, "Coordinates_" + ax + "_filtered")
        print("filtered data will be saved here:", savepath)
        
        hf.create_path_if_not_existent(savepath)  # Create save path if not exists

        # Loop through files in the specified folder
        for data_name in os.listdir(folder_name):
            if certain_subjects == []:
                certain_subjects = ["_"]

            # Check if the data name contains any of the specified subject identifiers
            if any(string in data_name for string in certain_subjects):

                # If override is False, check if coordinates already exist and skip this video
                if not override and os.path.exists(os.path.join(savepath, data_name)):
                    print("File " + data_name + " exists, skipping")
                    continue

                print("Filename:", data_name)

                # Load prepared coordinates from CSV
                coordinates = pd.read_csv(os.path.join(folder_name, data_name), index_col=None)

                # Delete 'Unnamed: 0' column if it exists
                try:
                    coordinates = coordinates.drop(columns=['Unnamed: 0'])
                except KeyError:
                    pass

                # Get all columns except the 'frame' column for filtering
                cols_to_filter = [col for col in coordinates.columns if col != 'frame']

                # Apply median filter to the specified columns
                coordinates[cols_to_filter] = coordinates[cols_to_filter].rolling(window_size, center=True, min_periods=1).median()

                # Save the filtered coordinates to a new CSV file
                coordinates.to_csv(os.path.join(savepath, data_name))

                
                
                
                
                
                

def parse_transmat_string(string):
    """
    Parses a string representing a transformation matrix into a 4x4 NumPy array.

    Args:
        string (str): The input string containing the transformation matrix.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the parsed transformation matrix.
    """
    # Preprocess the string to remove square brackets and split into lines
    lines = string.strip().replace('[', '').replace(']', '').split('\n')

    # Extract numeric values from each line and convert to float
    numeric_elements = [float(e) for line in lines for e in line.split()]

    # Validate the number of elements
    if len(numeric_elements) != 16:
        raise ValueError("Input string does not contain 16 numeric elements.")

    # Reshape elements into a 4x4 array
    array = np.array(numeric_elements).reshape(4, 4)

    return array
    
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

            hf.create_path_if_not_existent(X_savepath)
            X.to_csv(os.path.join(X_savepath, data_name))

            hf.create_path_if_not_existent(Y_savepath)
            Y.to_csv(os.path.join(Y_savepath, data_name))

            hf.create_path_if_not_existent(Z_savepath)
            Z.to_csv(os.path.join(Z_savepath, data_name))

            hf.create_path_if_not_existent(T_savepath)
            np.save(os.path.join(T_savepath, data_name + ".npy"), T)
            
            
            
            
            
def downsample_all_50_to_25_fps(folder, certain_subjects,override):

    fps = pd.read_csv('/home/ronjaschappert/Documents/UrgeTicAnalysis_2021/DeepTic/Tic_Rating_analysis/RUSH/final/data/RUSH/fps.csv')
    fps50 = fps["id"][fps["fps"] == 50.0].reset_index(drop=True)
    fps50 = [a.split(".")[0] for a in fps50]
    print(fps50)


    savemat = os.path.join(folder, "Transformation_matrix_25fps/")
    saveshape = os.path.join(folder, "Blendshape_25fps/")

    hf.create_path_if_not_existent(savemat)
    hf.create_path_if_not_existent(saveshape)
    for ax in ["X","Y","Z"]:
        folder_name = os.path.join(folder, "Coordinates_" + ax + "_prepared")
        save_path = os.path.join(folder,"Coordinates_" + ax + "_25fps")

        hf.create_path_if_not_existent(save_path)

        #also delete every second transformation mat and every second blendshape:
        Tmatpath = os.path.join(folder, "Transformation_matrix")
        Bshapepath = os.path.join(folder, "Blendshape")

        for data_name in os.listdir(folder_name):
             # if there are no certain subjects for which the face mesh should be applied, certain_subjects will be set to "_" so all subjects will be used
            if certain_subjects == []:
                certain_subjects = ["_"]
            if any(string in data_name for string in certain_subjects):
                # if override is false, check if coordinates already exist and skip this video
                if override == False:
                    if os.path.exists(os.path.join(save_path, data_name)):
                        print("file " + data_name  + " exists, skipping")
                        continue

                print("data name", data_name)

                coordinates = pd.read_csv(os.path.join(folder_name, data_name))
                print("shape before",coordinates.shape)


                # when downsampling X data, also downsample Tmat and Blendshape
                if ax == "X":
                    tmat = pd.read_csv(os.path.join(Tmatpath, data_name))
                    bshape = pd.read_csv(os.path.join(Bshapepath, data_name))

                if data_name.split(".")[0] in fps50:
                    print("50fps")
                    print(coordinates.columns)
                    print("sum_before", coordinates.drop(columns ="frame").sum().sum())
                    coordinates = coordinates.iloc[::2,:]
                    print("sum_after", coordinates.drop(columns = "frame").sum().sum())
                    if ax == "X":
                        tmat = tmat.iloc[::2,:]
                        bshape = bshape.iloc[::2,:]


                print("shape after downsampling", coordinates.shape)
                print("saving under", os.path.join(save_path,data_name))
                coordinates.to_csv(os.path.join(save_path,data_name))
                if ax == "X":
                    tmat.to_csv(os.path.join(savemat, data_name))
                    bshape.to_csv(os.path.join(saveshape, data_name))
                    
                    
                    
                    
def substract_one_point(coordinates):
        #substract coord of 0 (middle of upper lip))
        coordinates_copy = coordinates.copy()

        coordinates_copy = coordinates_copy.sub(coordinates["0"],axis = 0)
        coordinates_copy["frame"] = coordinates["frame"]
        
        return coordinates_copy
    
def normalize_face_mesh(X, Y, Z):
    #this function normalizes the face mesh coordinates

    # compute the mean of means and the mean of standard deviation of the frames
    b = ["X","Y","Z"]
    i = 0
    for ax in [X,Y,Z]:

        ax = ax.loc[:,~ax.columns.str.contains('Unnamed', case=False)] 
        ax = ax.loc[:,~ax.columns.str.contains('frame', case=False)] 


        ax = ax.assign(std = ax.std(axis=1))
        ax = ax.assign(mean = ax.mean(axis=1))

        #calc mean of means and stds:
        m = np.mean(ax[ax.sum(axis=1)!=0]["mean"])
        std1 = np.mean(ax[ax.sum(axis=1)!=0]["std"])
        ax.name = b[i]

        c = str(ax.name)+"_norm"

        ax = ax.drop(columns = ["std","mean"])      
        normalized = []

        ax[ax.sum(axis=1)!=0] = ax[ax.sum(axis=1)!=0].applymap(lambda x: (x-m)/std1)

        normalized = ax

        globals()[c] = normalized

        i+=1

    return (X_norm, Y_norm, Z_norm)

def center_and_normalize_data(folder, certain_subjects, override, center = True):
    if center:
        savefolder_X = os.path.join(folder,"Coordinates_X_centered_normalized")
        savefolder_Y = os.path.join(folder,"Coordinates_Y_centered_normalized")
        savefolder_Z = os.path.join(folder,"Coordinates_Z_centered_normalized")
    else:
        savefolder_X = os.path.join(folder,"Coordinates_X_normalized")
        savefolder_Y = os.path.join(folder,"Coordinates_Y_normalized")
        savefolder_Z = os.path.join(folder,"Coordinates_Z_normalized")

    for filename in os.listdir(os.path.join(folder, "Coordinates_X_standard")):
        if certain_subjects == []:
            certain_subjects = ["_"]
        if any(string in filename for string in certain_subjects):
            print("Filename:", filename)

            # If override is False, check if prepared coordinates already exist and skip this video
            if not override and os.path.exists(os.path.join(savefolder_X, filename)):
                print("File " + filename + " exists, skipping")
                continue

            print("filename", filename)
            X = pd.read_csv(os.path.join(folder, "Coordinates_X_standard/" +  filename))
            Y = pd.read_csv(os.path.join(folder, "Coordinates_Y_standard/" +  filename))
            Z = pd.read_csv(os.path.join(folder, "Coordinates_Z_standard/" +  filename))

            if center:
                X = substract_one_point(X)
                Y = substract_one_point(Y)
                Z = substract_one_point(Z)

            XX,YY,ZZ = normalize_face_mesh(X,Y,Z)
            for i in range(0,600,100):
                try:
                    plt.scatter(XX.iloc[i],-YY.iloc[i], s = 0.2)
                except:
                    pass
            plt.show()

            hf.create_path_if_not_existent(savefolder_X)
            hf.create_path_if_not_existent(savefolder_Y)
            hf.create_path_if_not_existent(savefolder_Z)

            XX.to_csv(os.path.join(savefolder_X, filename))
            YY.to_csv(os.path.join(savefolder_Y, filename))
            ZZ.to_csv(os.path.join(savefolder_Z, filename))
            
            
            
            
def downsample_all_50_to_25_fps_hype(folder, certain_subjects,override):



    savemat = os.path.join(folder, "Transformation_matrix_25fps/")
    saveshape = os.path.join(folder, "Blendshape_25fps/")

    hf.create_path_if_not_existent(savemat)
    hf.create_path_if_not_existent(saveshape)
    for ax in ["X","Y","Z"]:
        folder_name = os.path.join(folder, "Coordinates_" + ax + "_prepared")
        save_path = os.path.join(folder,"Coordinates_" + ax + "_25fps")

        hf.create_path_if_not_existent(save_path)

        #also delete every second transformation mat and every second blendshape:
        Tmatpath = os.path.join(folder, "Transformation_matrix")
        Bshapepath = os.path.join(folder, "Blendshape")

        for data_name in os.listdir(folder_name):
             # if there are no certain subjects for which the face mesh should be applied, certain_subjects will be set to "_" so all subjects will be used
            if certain_subjects == []:
                certain_subjects = ["_"]
            if any(string in data_name for string in certain_subjects):
                # if override is false, check if coordinates already exist and skip this video
                if override == False:
                    if os.path.exists(os.path.join(save_path, data_name)):
                        print("file " + data_name  + " exists, skipping")
                        continue

                print("data name", data_name)

                coordinates = pd.read_csv(os.path.join(folder_name, data_name))
                print("shape before",coordinates.shape)
        
                # when downsampling X data, also downsample Tmat and Blendshape
                if ax == "X":
                    tmat = pd.read_csv(os.path.join(Tmatpath, data_name))
                    bshape = pd.read_csv(os.path.join(Bshapepath, data_name))

                if len(coordinates) >7000:
                    print("50fps")
                    print(coordinates.columns)
                    print("sum_before", coordinates.drop(columns ="frame").sum().sum())
                    coordinates = coordinates.iloc[::2,:]
                    print("sum_after", coordinates.drop(columns = "frame").sum().sum())
                    if ax == "X":
                        tmat = tmat.iloc[::2,:]
                        bshape = bshape.iloc[::2,:]
                elif len(coordinates) ==3750:
                    print("25fps")
                else:
                    print("unknown fps", coordinates.shape)
                    return


                print("shape after downsampling", coordinates.shape)
                print("saving under", os.path.join(save_path,data_name))
                coordinates.to_csv(os.path.join(save_path,data_name))
                if ax == "X":
                    tmat.to_csv(os.path.join(savemat, data_name))
                    bshape.to_csv(os.path.join(saveshape, data_name))
                
                