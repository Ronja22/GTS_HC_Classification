from multiprocessing.spawn import prepare
import pandas as pd
import numpy as np
import os


import src.utils.directories as dir

  
    

def prepare_face_mesh_urge(input_folder, ratings_folder, certain_subjects=[], override=False):
    """
    Prepare face mesh data by converting and saving it in a structured format.

    Args:
        input_folder (str): The path to the main folder containing raw face mesh data.
        ratings_folder (str): The path to the folder containing ratings data.
        certain_subjects (list, optional): List of specific subjects to process. Default is an empty list.
        override (bool, optional): If False, skip processing for existing files. Default is False.

    Returns:
        None
    """
    
    # Iterate over the coordinate axes X, Y, Z
    for ax in ["X", "Y", "Z"]:
        # Define the input and output directories for the current axis
        startfolder = os.path.join(input_folder, "Coordinates_" + ax)
        savefolder = os.path.join(input_folder, "Coordinates_" + ax + "_prepared")
        
        # Create the output directory if it does not exist
        dir.create_directory_if_not_exists(savefolder)

        # Iterate over all files in the input directory
        for filename in os.listdir(startfolder):
            # If certain_subjects is empty, process all files
            if not certain_subjects:
                certain_subjects = ["_"]
            
            # Process files that match any of the certain_subjects strings
            if any(subject in filename for subject in certain_subjects):
                print("Processing filename:", filename)

                # Define the output file path
                output_file = os.path.join(savefolder, filename.split(".")[0] + ".csv")
                
                # If override is False and the output file already exists, skip processing
                if not override and os.path.exists(output_file):
                    print("File " + filename + " exists, skipping")
                    continue

                # Extract the base filename without extension
                filename = os.path.basename(filename).split(".")[0]
                

                # Define the paths for the transformation matrix and blendshapes
                transformation_path = os.path.join(input_folder, "Transformation_matrix", filename + ".csv")
                blendshape_path = os.path.join(input_folder, "Blendshape", filename + ".csv")
                
                # Load the raw face mesh data from the .npy file
                mesh = np.load(os.path.join(startfolder, filename + ".npy"))

                # Get the number of features and frames from the mesh data
                num_feat = mesh.shape[1]
                num_frames = mesh.shape[0]
                print("Number of features:", num_feat)
                print("Number of frames:", num_frames)

                # Convert the mesh data into a DataFrame
                mesh_df = pd.DataFrame(mesh, columns=range(num_feat))
                mesh_df["frame"] = range(num_frames)

                # Adjust filename for special cases (e.g., P3a)
                savename = filename
                if "P3a" in filename:
                    filename = filename.replace("UZL_E_", "").replace("_T1", "")
                    print("Adjusted filename for P3a:", filename)

                # Load the rating data
                rating_path = os.path.join(ratings_folder, filename + ".txt")
                rating = pd.read_csv(rating_path, delimiter=';')
                
                print("Length of rating data:", len(rating))
                
                # Calculate the start and end frames based on the ratings
                start_msec = rating.onset.iloc[0]
                end_msec = rating.onset.iloc[-1]

                # Convert milliseconds to frames (assuming 25 fps -> 0.025 frames per millisecond)
                start_frame = round(start_msec * 0.025)
                end_frame = round(end_msec * 0.025) + 24  # First frame of the last second + 24 more frames

                print("Frame range:", start_frame, "to", end_frame)

                # Ensure the frame column is numeric
                mesh_df["frame"] = pd.to_numeric(mesh_df["frame"])

                # Process and save transformation matrix and blendshapes 
                if ax == "X":
                    # Load and cut the transformation matrix
                    transformation = pd.read_csv(transformation_path)
                    transformation["frame"] = mesh_df["frame"]
                    transformation_cut = transformation.loc[transformation.frame.between(start_frame, end_frame)]
                    
                    # Load and cut the blendshapes
                    blendshape = pd.read_csv(blendshape_path)
                    blendshape["frame"] = mesh_df["frame"]
                    blendshape_cut = blendshape.loc[blendshape.frame.between(start_frame, end_frame)]
                
                    transformationsavepath = os.path.join(input_folder, "transformation_cut")
                    dir.create_directory_if_not_exists(transformationsavepath)
                    transformation_cut.to_csv(os.path.join(transformationsavepath, savename + ".csv"), index=False)
                    
                    blendshapesavepath = os.path.join(input_folder, "blendshape_cut")
                    dir.create_directory_if_not_exists(blendshapesavepath)
                    blendshape_cut.to_csv(os.path.join(blendshapesavepath, savename + ".csv"), index=False)
                
                # Cut the face mesh data for the current axis
                mesh_cut = mesh_df.loc[mesh_df.frame.between(start_frame, end_frame)]
                
                print("Length of cut mesh data:", len(mesh_cut))
                # Save the prepared face mesh data
                mesh_cut.to_csv(os.path.join(savefolder, savename + ".csv"), index=False)

                

def prepare_face_mesh_rush_hype(input_folder, certain_subjects=[], override=False):
    """
    Prepare face mesh data by converting and saving it in a structured format.

    Args:
        input_folder (str): The path to the main folder containing raw face mesh data.
        certain_subjects (list, optional): List of specific subjects to process. Default is an empty list.
        override (bool, optional): If False, skip processing for existing files. Default is False.

    Returns:
        None
    """
    
    # Iterate over the X, Y, Z coordinate directories
    for ax in ["X", "Y", "Z"]:
        startfolder = os.path.join(input_folder, "Coordinates_" + ax)
        savefolder = os.path.join(input_folder, "Coordinates_" + ax + "_prepared")
        
        # Create the output directory if it does not exist
        dir.create_directory_if_not_exists(savefolder)

        # Iterate over all files in the start folder
        for filename in os.listdir(startfolder):
            # If certain_subjects is empty, default to processing all files
            if not certain_subjects:
                certain_subjects = ["_"]
            
            # Process files that match any of the certain_subjects strings
            if any(subject in filename for subject in certain_subjects):
                print("Processing filename:", filename)

                
                output_file = os.path.join(savefolder, filename.split(".")[0] + ".csv")
                
                # If override is False, skip files that already exist in the save folder
                if not override and os.path.exists(output_file):
                    print("File " + filename + " exists, skipping")
                    continue

                # Load the raw face mesh data from the .npy file
                mesh = np.load(os.path.join(startfolder, filename))

                # Get the dimensions of the mesh data (num_frames x num_features)
                num_frames, num_features = mesh.shape

                # Convert mesh data to a DataFrame with frame numbers
                mesh_df = pd.DataFrame(mesh, columns=range(num_features))
                mesh_df["frame"] = range(num_frames)
                mesh_df["frame"] = pd.to_numeric(mesh_df["frame"])  # Ensure frame column is numeric

                # Save the prepared face mesh data to a CSV file in the save folder
                mesh_df.to_csv(output_file, index=False)
                print(f"Saved prepared data to {output_file}")


